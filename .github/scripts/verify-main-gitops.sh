#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
publisher="${repo_root}/.github/scripts/publish-main-image.sh"
workflow="${repo_root}/.github/workflows/prod-ci.yml"
main_workflow="${repo_root}/.github/workflows/main-ci.yml"
fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }
[[ -x "${publisher}" ]] || fail "missing executable publish-main-image.sh"
[[ -f "${workflow}" ]] || fail "missing prod-ci.yml"
[[ -f "${main_workflow}" ]] || fail "existing main-ci.yml must be preserved"
[[ "$(grep -Fc 'branches: [prod]' "${workflow}")" -eq 2 ]] || fail "PR and push production triggers must target prod"
! grep -Fq 'branches: [main]' "${workflow}" || fail "production workflow must not trigger from main"
[[ "$(grep -Fc 'branches: [main]' "${main_workflow}")" -eq 2 ]] || fail "main workflow must keep PR and push main triggers"
! grep -Fq 'branches: [prod]' "${main_workflow}" || fail "main workflow must not trigger from prod"
grep -Fq "github.ref == 'refs/heads/prod'" "${workflow}" || fail "image publication must require the prod ref"
! grep -Fq "github.ref == 'refs/heads/main'" "${workflow}" || fail "image publication must not run from main"
[[ "$(grep -Fc -- '--provenance=false' "${workflow}")" -eq 2 ]] || fail "prod PR and publish builds must disable default attestations"
[[ "$(grep -Fc -- '--provenance=false' "${main_workflow}")" -eq 1 ]] || fail "main build must disable default attestations"

test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT
fake_bin="${test_root}/bin"
mkdir -p "${fake_bin}"

cat >"${fake_bin}/docker" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf 'docker %s\n' "$*" >>"${CALL_LOG}"
FAKE
cat >"${fake_bin}/aws" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf 'aws %s\n' "$*" >>"${CALL_LOG}"
case "$*" in
  *"ecr wait image-scan-complete"*) exit "${SCAN_WAIT_EXIT:-0}" ;;
  *"ecr describe-image-scan-findings"*)
    if [[ -n "${SCAN_NOT_FOUND_ONCE_FILE:-}" && ! -e "${SCAN_NOT_FOUND_ONCE_FILE}" ]]; then
      : >"${SCAN_NOT_FOUND_ONCE_FILE}"
      printf 'ScanNotFoundException\n' >&2
      exit 255
    fi
    printf '%s\n' "${SCAN_RESPONSE:-{\"status\":\"COMPLETE\",\"findings\":{\"findingSeverityCounts\":{\"CRITICAL\":${CRITICAL_COUNT:-0}}}}}"
    ;;
  *"ecr describe-images"*"imageTag=approved-"*) printf '%s\n' "${EXISTING_APPROVED_DIGEST:-None}" ;;
  *"ecr describe-images"*"imageTag="*) printf '%s\n' "${PUSHED_DIGEST}" ;;
  *"ecr describe-images"*) printf '%s\n' "${APPROVED_IMAGES_JSON:-[]}" ;;
  *"ecr batch-get-image"*) printf '%s\n' '{"schemaVersion":2}' ;;
  *"ecr put-image"*) printf '%s\n' "${PUSHED_DIGEST}" ;;
esac
FAKE
chmod +x "${fake_bin}"/*

git_root="${test_root}/git"
git init -q "${git_root}"
git -C "${git_root}" config user.name test
git -C "${git_root}" config user.email test@example.com
printf 'base\n' >"${git_root}/state"
git -C "${git_root}" add state
git -C "${git_root}" commit -qm base
base_sha="$(git -C "${git_root}" rev-parse HEAD)"
printf 'forward\n' >"${git_root}/state"
git -C "${git_root}" commit -qam forward
forward_sha="$(git -C "${git_root}" rev-parse HEAD)"
printf 'latest\n' >"${git_root}/state"
git -C "${git_root}" commit -qam latest
latest_sha="$(git -C "${git_root}" rev-parse HEAD)"
git -C "${git_root}" checkout -qb diverged "${base_sha}"
printf 'diverged\n' >"${git_root}/state"
git -C "${git_root}" commit -qam diverged
diverged_sha="$(git -C "${git_root}" rev-parse HEAD)"

digest='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
other_digest='sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'

run_publish() {
  local candidate="$1"
  local call_log="$2"
  shift 2
  (
    cd "${git_root}"
    env PATH="${fake_bin}:${PATH}" CALL_LOG="${call_log}" \
      AWS_REGION=ap-northeast-2 AWS_ACCOUNT_ID=352635536602 \
      ECR_REPOSITORY=cosmos-v2/fastapi IMAGE_TAG="${candidate}" \
      LOCAL_IMAGE="ai-test:${candidate}" PUSHED_DIGEST="${digest}" \
      "$@" "${publisher}"
  )
}

success_log="${test_root}/success.log"
run_publish "${forward_sha}" "${success_log}"
grep -Fq "ecr describe-image-scan-findings" "${success_log}" || fail "must poll ECR Basic Scan findings"
! grep -Fq "ecr wait image-scan-complete" "${success_log}" || fail "must not use the non-retrying ECR waiter"
grep -Fq "ecr describe-image-scan-findings" "${success_log}" || fail "must read CRITICAL findings"
grep -Fq "ecr batch-get-image" "${success_log}" || fail "must fetch the scanned manifest"
grep -Fq "ecr put-image" "${success_log}" || fail "must add the approved tag"
grep -Fq -- "--image-tag approved-${forward_sha}" "${success_log}" || fail "must use approved-<SHA>"

scan_retry_marker="${test_root}/scan-not-found-once"
run_publish "${forward_sha}" "${test_root}/scan-retry.log" \
  SCAN_NOT_FOUND_ONCE_FILE="${scan_retry_marker}" ECR_SCAN_POLL_INTERVAL_SECONDS=0

run_publish "${forward_sha}" "${test_root}/forward.log" APPROVED_IMAGES_JSON="[{\"pushedAt\":\"2026-11-01T00:00:00Z\",\"tags\":[\"approved-${base_sha}\"]}]"

critical_log="${test_root}/critical.log"
if run_publish "${forward_sha}" "${critical_log}" CRITICAL_COUNT=1 >/dev/null 2>&1; then fail "must reject CRITICAL ECR findings"; fi
grep -Fq "ecr put-image" "${critical_log}" && fail "must not approve a CRITICAL image"

stale_log="${test_root}/stale.log"
if run_publish "${base_sha}" "${stale_log}" APPROVED_IMAGES_JSON="[{\"pushedAt\":\"2026-11-02T00:00:00Z\",\"tags\":[\"approved-${forward_sha}\"]}]" >/dev/null 2>&1; then fail "must reject a stale candidate"; fi
grep -Fq "ecr put-image" "${stale_log}" && fail "must not approve a stale candidate"

if run_publish "${diverged_sha}" "${test_root}/diverged.log" APPROVED_IMAGES_JSON="[{\"pushedAt\":\"2026-11-02T00:00:00Z\",\"tags\":[\"approved-${forward_sha}\"]}]" >/dev/null 2>&1; then fail "must reject a diverged candidate"; fi

rerun_log="${test_root}/rerun.log"
run_publish "${forward_sha}" "${rerun_log}" APPROVED_IMAGES_JSON="[{\"pushedAt\":\"2026-11-02T00:00:00Z\",\"tags\":[\"approved-${forward_sha}\"]}]" EXISTING_APPROVED_DIGEST="${digest}"
grep -Fq "ecr put-image" "${rerun_log}" && fail "same-digest rerun must be a no-op"

if run_publish "${forward_sha}" "${test_root}/mismatch.log" APPROVED_IMAGES_JSON="[{\"pushedAt\":\"2026-11-02T00:00:00Z\",\"tags\":[\"approved-${forward_sha}\"]}]" EXISTING_APPROVED_DIGEST="${other_digest}" >/dev/null 2>&1; then fail "must reject an approved tag that points at another digest"; fi
if run_publish "${forward_sha}" "${test_root}/invalid.log" CRITICAL_COUNT=unknown >/dev/null 2>&1; then fail "must fail closed when scan counts are invalid"; fi
if run_publish "${forward_sha}" "${test_root}/missing-findings.log" SCAN_RESPONSE='{"status":"COMPLETE","findings":null}' >/dev/null 2>&1; then fail "must fail closed when scan findings are missing"; fi
if run_publish "${forward_sha}" "${test_root}/failed-scan.log" SCAN_RESPONSE='{"status":"FAILED","findings":{"findingSeverityCounts":{}}}' >/dev/null 2>&1; then fail "must fail closed when scan status is not COMPLETE"; fi
paginated_log="${test_root}/paginated.log"
if run_publish "${forward_sha}" "${paginated_log}" APPROVED_IMAGES_JSON="[{\"pushedAt\":\"2026-11-01T00:00:00Z\",\"tags\":[\"approved-${base_sha}\"]},{\"pushedAt\":\"2026-11-03T00:00:00Z\",\"tags\":[\"approved-${latest_sha}\"]}]" >/dev/null 2>&1; then fail "must select the globally newest approved tag across paginated results"; fi
grep -Fq "ecr put-image" "${paginated_log}" && fail "must not approve a candidate older than a later-page approval"
if run_publish bad "${test_root}/bad.log" >/dev/null 2>&1; then fail "must reject a non-SHA image tag"; fi

printf 'PASS: AI approved-image publication contract\n'
