#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build-docker.sh [cpu|gpu|all] [options]

Build FunSpeech Docker images on a server and optionally export them as tarballs.

Options:
  --image NAME      Image repository/name. Default: funspeech
  --tag TAG         CPU image tag. Default: latest
  --gpu-tag TAG     GPU image tag. Default: gpu-latest
  --output DIR      Tarball output directory. Default: dist
  --no-save         Only build images; do not write docker save tarballs
  --push            Push built images after build
  --no-cache        Build without Docker layer cache
  -h, --help        Show this help

Examples:
  scripts/build-docker.sh cpu
  scripts/build-docker.sh gpu --image docker.cnb.cool/nexa/funspeech --push
  TAG=v1.2.0 scripts/build-docker.sh all --output /tmp/funspeech-images
EOF
}

variant="cpu"
image_name="${IMAGE_NAME:-funspeech}"
cpu_tag="${TAG:-latest}"
gpu_tag="${GPU_TAG:-gpu-latest}"
output_dir="${OUTPUT_DIR:-dist}"
save_image="${SAVE_IMAGE:-1}"
push_image="${PUSH_IMAGE:-0}"
no_cache="${NO_CACHE:-0}"

if [[ $# -gt 0 && "$1" != --* ]]; then
  variant="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      image_name="${2:?--image requires a value}"
      shift 2
      ;;
    --tag)
      cpu_tag="${2:?--tag requires a value}"
      shift 2
      ;;
    --gpu-tag)
      gpu_tag="${2:?--gpu-tag requires a value}"
      shift 2
      ;;
    --output)
      output_dir="${2:?--output requires a value}"
      shift 2
      ;;
    --no-save)
      save_image="0"
      shift
      ;;
    --push)
      push_image="1"
      shift
      ;;
    --no-cache)
      no_cache="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$variant" in
  cpu|gpu|all) ;;
  *)
    echo "Unknown variant: $variant. Expected cpu, gpu, or all." >&2
    exit 2
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but was not found in PATH." >&2
  exit 127
fi

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

build_args=()
if [[ "$no_cache" == "1" ]]; then
  build_args+=(--no-cache)
fi

safe_name="${image_name//\//_}"
safe_name="${safe_name//:/_}"

build_one() {
  local label="$1"
  local dockerfile="$2"
  local tag="$3"
  local image_ref="${image_name}:${tag}"

  echo "==> Building ${label} image: ${image_ref}"
  docker build "${build_args[@]}" -t "$image_ref" -f "$dockerfile" .

  if [[ "$save_image" == "1" ]]; then
    mkdir -p "$output_dir"
    local tarball="${output_dir}/${safe_name}-${tag}.tar.gz"
    echo "==> Saving ${image_ref} to ${tarball}"
    docker save "$image_ref" | gzip -c > "$tarball"
  fi

  if [[ "$push_image" == "1" ]]; then
    echo "==> Pushing ${image_ref}"
    docker push "$image_ref"
  fi
}

case "$variant" in
  cpu)
    build_one "cpu" "Dockerfile" "$cpu_tag"
    ;;
  gpu)
    build_one "gpu" "Dockerfile.gpu" "$gpu_tag"
    ;;
  all)
    build_one "cpu" "Dockerfile" "$cpu_tag"
    build_one "gpu" "Dockerfile.gpu" "$gpu_tag"
    ;;
esac

echo "==> Done."
