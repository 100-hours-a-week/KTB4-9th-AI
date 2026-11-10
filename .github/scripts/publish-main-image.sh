#!/usr/bin/env bash
set -euo pipefail

required=(AWS_REGION AWS_ACCOUNT_ID ECR_REPOSITORY IMAGE_TAG LOCAL_IMAGE)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "${name}" >&2; exit 1; }
done

[[ "${IMAGE_TAG}" =~ ^[0-9a-f]{40}$ ]] || { printf 'ERROR: IMAGE_TAG must be a 40-character lowercase Git SHA\n' >&2; exit 1; }
[[ "${ECR_REPOSITORY}" == "cosmos-v2/fastapi" ]] || { printf 'ERROR: ECR_REPOSITORY must be cosmos-v2/fastapi\n' >&2; exit 1; }

registry="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
remote_image="${registry}/${ECR_REPOSITORY}:${IMAGE_TAG}"
approved_tag="approved-${IMAGE_TAG}"

docker tag "${LOCAL_IMAGE}" "${remote_image}"
docker push "${remote_image}"

scan_poll_interval="${ECR_SCAN_POLL_INTERVAL_SECONDS:-5}"
scan_max_attempts="${ECR_SCAN_MAX_ATTEMPTS:-60}"
[[ "${scan_poll_interval}" =~ ^[0-9]+$ && "${scan_max_attempts}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'ERROR: invalid ECR scan polling configuration\n' >&2
  exit 1
}

scan_result=""
scan_complete=false
scan_error="$(mktemp)"
trap 'rm -f "${scan_error}"' EXIT
for ((attempt = 1; attempt <= scan_max_attempts; attempt++)); do
  if scan_result="$(aws ecr describe-image-scan-findings \
    --repository-name "${ECR_REPOSITORY}" \
    --image-id "imageTag=${IMAGE_TAG}" \
    --region "${AWS_REGION}" \
    --query '{status:imageScanStatus.status,findings:imageScanFindings}' \
    --output json 2>"${scan_error}")"; then
    scan_status="$(jq -r '.status // empty' <<<"${scan_result}")"
    case "${scan_status}" in
      COMPLETE) scan_complete=true; break ;;
      IN_PROGRESS) ;;
      *) printf 'ERROR: ECR image scan entered unexpected status: %s\n' "${scan_status:-missing}" >&2; exit 1 ;;
    esac
  elif ! grep -Fq 'ScanNotFoundException' "${scan_error}"; then
    cat "${scan_error}" >&2
    exit 1
  fi

  sleep "${scan_poll_interval}"
done
rm -f "${scan_error}"
trap - EXIT
[[ "${scan_complete}" == true ]] || { printf 'ERROR: ECR image scan did not complete in time\n' >&2; exit 1; }

critical="$(jq -er '
  if .status != "COMPLETE" then error("scan is not complete")
  elif (.findings | type) != "object" then error("scan findings are missing")
  else (.findings.findingSeverityCounts.CRITICAL // 0)
  end
' <<<"${scan_result}")" || { printf 'ERROR: ECR scan result is incomplete or invalid\n' >&2; exit 1; }
[[ "${critical}" =~ ^[0-9]+$ ]] || { printf 'ERROR: ECR scan did not return a CRITICAL count\n' >&2; exit 1; }
(( critical == 0 )) || { printf 'ERROR: ECR Basic Scan found %s CRITICAL vulnerabilities\n' "${critical}" >&2; exit 1; }

digest="$(aws ecr describe-images \
  --repository-name "${ECR_REPOSITORY}" --image-ids "imageTag=${IMAGE_TAG}" --region "${AWS_REGION}" \
  --query 'imageDetails[0].imageDigest' --output text)"
[[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || { printf 'ERROR: pushed image digest was not found\n' >&2; exit 1; }

approved_images="$(aws ecr describe-images \
  --repository-name "${ECR_REPOSITORY}" --region "${AWS_REGION}" \
  --query 'imageDetails[].{pushedAt:imagePushedAt,tags:imageTags}' --output json)"
latest_approved="$(jq -er '
  if type != "array" then error("image details are not an array")
  else [
    .[] as $image
    | select(($image.pushedAt | type) == "string")
    | ($image.tags // [])[]
    | select(type == "string" and test("^approved-[0-9a-f]{40}$"))
    | {tag: ., pushedAt: $image.pushedAt}
  ] | sort_by(.pushedAt) | if length == 0 then "" else .[-1].tag end
  end
' <<<"${approved_images}")" || { printf 'ERROR: approved image inventory is invalid\n' >&2; exit 1; }
if [[ -n "${latest_approved}" && "${latest_approved}" != "None" ]]; then
  [[ "${latest_approved}" =~ ^approved-([0-9a-f]{40})$ ]] || { printf 'ERROR: latest approved tag is invalid\n' >&2; exit 1; }
  current_sha="${BASH_REMATCH[1]}"
  git cat-file -e "${current_sha}^{commit}" 2>/dev/null || { printf 'ERROR: current approved SHA is missing from Git history\n' >&2; exit 1; }
  git merge-base --is-ancestor "${current_sha}" "${IMAGE_TAG}" || { printf 'ERROR: candidate SHA is stale or diverged from the approved deployment\n' >&2; exit 1; }
fi

existing_digest="$(aws ecr describe-images \
  --repository-name "${ECR_REPOSITORY}" --image-ids "imageTag=${approved_tag}" --region "${AWS_REGION}" \
  --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)"
if [[ -n "${existing_digest}" && "${existing_digest}" != "None" ]]; then
  [[ "${existing_digest}" == "${digest}" ]] || { printf 'ERROR: approved tag already points at a different digest\n' >&2; exit 1; }
  printf 'Approved image already exists: %s@%s\n' "${registry}/${ECR_REPOSITORY}" "${digest}"
  exit 0
fi

manifest="$(aws ecr batch-get-image \
  --repository-name "${ECR_REPOSITORY}" --image-ids "imageTag=${IMAGE_TAG}" --region "${AWS_REGION}" \
  --query 'images[0].imageManifest' --output text)"
[[ -n "${manifest}" && "${manifest}" != "None" ]] || { printf 'ERROR: source image manifest was not found\n' >&2; exit 1; }
approved_digest="$(aws ecr put-image \
  --repository-name "${ECR_REPOSITORY}" --image-tag "${approved_tag}" --image-manifest "${manifest}" --region "${AWS_REGION}" \
  --query 'image.imageId.imageDigest' --output text)"
[[ "${approved_digest}" == "${digest}" ]] || { printf 'ERROR: approved tag digest does not match the scanned image\n' >&2; exit 1; }

printf 'Published and approved %s:%s at %s\n' "${registry}/${ECR_REPOSITORY}" "${approved_tag}" "${digest}"
