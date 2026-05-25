#!/usr/bin/env bash
# Verify that the host has the expected toolchain installed.

set -euo pipefail

echo "Python:        $(python --version 2>&1)"
echo "pip:           $(pip --version)"
echo "Package:       $(python -c 'import llm_memory_eval; print(llm_memory_eval.__version__)' 2>&1)"

echo
echo "Optional dependencies:"
for pkg in numpy pandas scipy pydantic sentence_transformers faiss matplotlib together boto3; do
  if python -c "import importlib; importlib.import_module(\"$pkg\")" 2>/dev/null; then
    version=$(python -c "import importlib; m = importlib.import_module(\"$pkg\"); print(getattr(m, '__version__', 'unknown'))")
    printf "  %-22s %s\n" "$pkg" "$version"
  else
    printf "  %-22s MISSING\n" "$pkg"
  fi
done

echo
echo "Backend environment:"
for var in TOGETHER_API_KEY OPENAI_API_KEY AWS_PROFILE AWS_REGION HF_TOKEN OLLAMA_URL; do
  if [ -n "${!var:-}" ]; then
    printf "  %-22s set\n" "$var"
  else
    printf "  %-22s unset\n" "$var"
  fi
done
