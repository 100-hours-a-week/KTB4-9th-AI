#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
publisher="${repo_root}/.github/scripts/publish-main-image.sh"
fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }
[[ -x "${publisher}" ]] || fail "missing executable publish-main-image.sh"

test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT
mkdir -p "${test_root}/bin"
call_log="${test_root}/calls.log"

cat >"${test_root}/bin/docker" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf 'docker %s\n' "$*" >>"${CALL_LOG}"
FAKE
cat >"${test_root}/bin/aws" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf 'aws %s\n' "$*" >>"${CALL_LOG}"
[[ "$*" == *"ecr describe-images"* ]] && printf 'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n'
FAKE
cat >"${test_root}/bin/gh" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf 'gh %s\n' "$*" >>"${CALL_LOG}"
FAKE
chmod +x "${test_root}/bin"/*

sha="fedcba9876543210fedcba9876543210fedcba98"
PATH="${test_root}/bin:${PATH}" CALL_LOG="${call_log}" \
AWS_REGION=ap-northeast-2 AWS_ACCOUNT_ID=352635536602 ECR_REPOSITORY=cosmos-v2/fastapi \
IMAGE_TAG="${sha}" LOCAL_IMAGE="ai-test:${sha}" COMPONENT=fastapi SOURCE_REPOSITORY=cosmos-v2/fastapi \
SOURCE_REPO_SLUG=100-hours-a-week/KTB4-9th-AI CLOUD_REPOSITORY=100-hours-a-week/KTB4-9th-Cloud \
GH_TOKEN=test-token "${publisher}"

call1="$(sed -n '1p' "${call_log}")"
call2="$(sed -n '2p' "${call_log}")"
call3="$(sed -n '3p' "${call_log}")"
call4="$(sed -n '4p' "${call_log}")"
[[ "${call1}" == "docker tag ai-test:${sha} 352635536602.dkr.ecr.ap-northeast-2.amazonaws.com/cosmos-v2/fastapi:${sha}" ]] || fail "must tag the already-scanned AI image"
[[ "${call2}" == "docker push 352635536602.dkr.ecr.ap-northeast-2.amazonaws.com/cosmos-v2/fastapi:${sha}" ]] || fail "must push the immutable SHA image"
[[ "${call3}" == *"imageTag=${sha}"* ]] || fail "must resolve the pushed image digest"
[[ "${call4}" == *"event_type=promote-v3"* ]] || fail "must emit promote-v3"
[[ "${call4}" == *"client_payload[component]=fastapi"* ]] || fail "must identify fastapi"
[[ "${call4}" == *"client_payload[repository]=cosmos-v2/fastapi"* ]] || fail "must identify the source ECR repository"
[[ "${call4}" == *"client_payload[image_tag]=${sha}"* ]] || fail "must forward the source revision"
[[ "${call4}" == *"client_payload[source_repository]=100-hours-a-week/KTB4-9th-AI"* ]] || fail "must identify the AI repository"

if PATH="${test_root}/bin:${PATH}" CALL_LOG="${call_log}" AWS_REGION=ap-northeast-2 AWS_ACCOUNT_ID=352635536602 ECR_REPOSITORY=cosmos-v2/fastapi IMAGE_TAG=bad LOCAL_IMAGE=test COMPONENT=fastapi SOURCE_REPOSITORY=cosmos-v2/fastapi SOURCE_REPO_SLUG=100-hours-a-week/KTB4-9th-AI CLOUD_REPOSITORY=100-hours-a-week/KTB4-9th-Cloud GH_TOKEN=test "${publisher}" >/dev/null 2>&1; then
  fail "must reject a non-SHA image tag"
fi

printf 'PASS: AI main GitOps publication contract\n'
