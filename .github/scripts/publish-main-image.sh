#!/usr/bin/env bash
set -euo pipefail

required=(AWS_REGION AWS_ACCOUNT_ID ECR_REPOSITORY IMAGE_TAG LOCAL_IMAGE COMPONENT SOURCE_REPOSITORY SOURCE_REPO_SLUG CLOUD_REPOSITORY GH_TOKEN)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { printf 'ERROR: missing %s\n' "${name}" >&2; exit 1; }
done
[[ "${IMAGE_TAG}" =~ ^[0-9a-f]{40}$ ]] || { printf 'ERROR: IMAGE_TAG must be a 40-character lowercase Git SHA\n' >&2; exit 1; }
[[ "${ECR_REPOSITORY}" == "${SOURCE_REPOSITORY}" ]] || { printf 'ERROR: source ECR repository mismatch\n' >&2; exit 1; }

registry="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
remote_image="${registry}/${ECR_REPOSITORY}:${IMAGE_TAG}"
docker tag "${LOCAL_IMAGE}" "${remote_image}"
docker push "${remote_image}"

digest="$(aws ecr describe-images --repository-name "${ECR_REPOSITORY}" --image-ids "imageTag=${IMAGE_TAG}" --region "${AWS_REGION}" --query 'imageDetails[0].imageDigest' --output text)"
[[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || { printf 'ERROR: pushed image digest was not found\n' >&2; exit 1; }

gh api --method POST "repos/${CLOUD_REPOSITORY}/dispatches" \
  -f 'event_type=promote-v3' \
  -f "client_payload[component]=${COMPONENT}" \
  -f "client_payload[repository]=${SOURCE_REPOSITORY}" \
  -f "client_payload[image_tag]=${IMAGE_TAG}" \
  -f "client_payload[source_repository]=${SOURCE_REPO_SLUG}"
printf 'Published %s@%s and requested Cloud promotion\n' "${registry}/${ECR_REPOSITORY}" "${digest}"
