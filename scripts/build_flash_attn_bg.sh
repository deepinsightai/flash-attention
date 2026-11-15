#!/usr/bin/env bash
# Wrapper so the flash-attn build can be launched detached from Windows:
#   Start-Process wsl.exe -ArgumentList '-d','Ubuntu-24.04','--','bash','/mnt/c/.../scripts/build_flash_attn_bg.sh' -WindowStyle Hidden
cd "$(dirname "$0")/.." || exit 1
sed -i 's/\r$//' scripts/*.sh
bash scripts/build_flash_attn.sh > "$HOME/flash_attn_build.log" 2>&1
echo "BUILD_DONE exit=$?" >> "$HOME/flash_attn_build.log"
