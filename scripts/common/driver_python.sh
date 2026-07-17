#!/usr/bin/env bash

DRIVER_PYTHON_MIN_VERSION="3.9"

_driver_python_command_path() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] || return 1
    printf '%s\n' "${candidate}"
    return 0
  fi
  command -v -- "${candidate}" 2>/dev/null
}

_driver_python_version() {
  "$1" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null
}

_driver_python_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null
}

resolve_driver_python() {
  if [[ "${DRIVER_PYTHON_RESOLVED:-0}" == "1" \
    && -n "${DRIVER_PYTHON:-}" \
    && -n "${DRIVER_PYTHON_VERSION:-}" ]] \
    && _driver_python_supported "${DRIVER_PYTHON}"; then
    return 0
  fi
  unset DRIVER_PYTHON_RESOLVED DRIVER_PYTHON_VERSION

  local explicit=0
  local home_dir="${HOME:-}"
  local requested="${DRIVER_PYTHON:-}"
  local -a candidates=()
  if [[ -n "${requested}" ]]; then
    explicit=1
    candidates+=("${requested}")
  else
    candidates+=(python3)
    [[ -n "${CONDA_PREFIX:-}" ]] && candidates+=("${CONDA_PREFIX}/bin/python3")
    candidates+=(python3.13 python3.12 python3.11 python3.10 python3.9)
    if [[ -n "${home_dir}" ]]; then
      candidates+=("${home_dir}/miniconda3/bin/python3" "${home_dir}/anaconda3/bin/python3")
    fi
    candidates+=(/opt/conda/bin/python3 /opt/miniconda3/bin/python3)
  fi

  local candidate resolved version
  local -a checked=()
  declare -A seen=()
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if ! resolved="$(_driver_python_command_path "${candidate}")"; then
      checked+=("${candidate}=not-found")
      continue
    fi
    [[ -z "${seen[${resolved}]:-}" ]] || continue
    seen["${resolved}"]=1
    version="$(_driver_python_version "${resolved}" || true)"
    if _driver_python_supported "${resolved}"; then
      DRIVER_PYTHON="${resolved}"
      DRIVER_PYTHON_VERSION="${version}"
      DRIVER_PYTHON_RESOLVED=1
      export DRIVER_PYTHON DRIVER_PYTHON_VERSION DRIVER_PYTHON_RESOLVED
      return 0
    fi
    checked+=("${resolved}=${version:-unusable}")
  done

  if [[ "${explicit}" == "1" ]]; then
    echo "[driver-python] DRIVER_PYTHON=${requested} is unavailable or does not satisfy Python >=${DRIVER_PYTHON_MIN_VERSION}." >&2
  else
    echo "[driver-python] no Python >=${DRIVER_PYTHON_MIN_VERSION} interpreter was found on the developer machine." >&2
  fi
  if (( ${#checked[@]} > 0 )); then
    printf '[driver-python] checked: %s\n' "${checked[*]}" >&2
  fi
  echo "[driver-python] set DRIVER_PYTHON=/path/to/python3.9-or-newer." >&2
  return 2
}

print_driver_python() {
  printf '[driver-python] executable: %s\n' "${DRIVER_PYTHON}"
  printf '[driver-python] version   : %s\n' "${DRIVER_PYTHON_VERSION}"
}
