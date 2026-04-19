#!/usr/bin/env bash
# tools/vendor-headers.sh
set -euo pipefail

XTL_VERSION="0.7.7"
XTENSOR_VERSION="0.25.0"
XAD_VERSION="2.1.0"

INST_INCLUDE="$(cd "$(dirname "$0")/.." && pwd)/inst/include"

echo "==> Vendoring headers into ${INST_INCLUDE}"

# xtl
XTL_URL="https://github.com/xtensor-stack/xtl/archive/refs/tags/${XTL_VERSION}.tar.gz"
XTL_DIR="${INST_INCLUDE}/xtl"
echo "--> Downloading xtl ${XTL_VERSION}..."
mkdir -p "${XTL_DIR}"
curl -fsSL "${XTL_URL}" \
  | tar -xz --strip-components=2 -C "${XTL_DIR}" \
      "xtl-${XTL_VERSION}/include/xtl" \
  || { echo "ERROR: failed to vendor xtl ${XTL_VERSION}" >&2; exit 1; }
echo "    xtl headers installed to ${XTL_DIR}"

# xtensor
XTENSOR_URL="https://github.com/xtensor-stack/xtensor/archive/refs/tags/${XTENSOR_VERSION}.tar.gz"
XTENSOR_DIR="${INST_INCLUDE}/xtensor"
echo "--> Downloading xtensor ${XTENSOR_VERSION}..."
mkdir -p "${XTENSOR_DIR}"
curl -fsSL "${XTENSOR_URL}" \
  | tar -xz --strip-components=2 -C "${XTENSOR_DIR}" \
      "xtensor-${XTENSOR_VERSION}/include/xtensor" \
  || { echo "ERROR: failed to vendor xtensor ${XTENSOR_VERSION}" >&2; exit 1; }
echo "    xtensor headers installed to ${XTENSOR_DIR}"

# XAD
XAD_URL="https://github.com/auto-differentiation/xad/archive/refs/tags/v${XAD_VERSION}.tar.gz"
XAD_DIR="${INST_INCLUDE}/XAD"
echo "--> Downloading XAD ${XAD_VERSION}..."
mkdir -p "${XAD_DIR}"
curl -fsSL "${XAD_URL}" \
  | tar -xz --strip-components=2 -C "${XAD_DIR}" \
      "xad-${XAD_VERSION}/src/XAD" \
  || { echo "ERROR: failed to vendor XAD ${XAD_VERSION}" >&2; exit 1; }
echo "    XAD headers installed to ${XAD_DIR}"

echo ""
echo "==> Done. Header counts:"
echo "    xtl:     $(find "${XTL_DIR}"     -name '*.hpp' | wc -l) files"
echo "    xtensor: $(find "${XTENSOR_DIR}" -name '*.hpp' | wc -l) files"
echo "    XAD:     $(find "${XAD_DIR}"     -name '*.hpp' | wc -l) files"
