#!/bin/bash
# parse_options.sh - 解析命令行参数

while [ $# -gt 0 ]; do
    case "$1" in
        --*=*)
            name="${1%%=*}"
            name="${name#--}"
            value="${1#*=}"
            eval "${name}=\"${value}\""
            ;;
        --*)
            name="${1#--}"
            shift
            value="$1"
            eval "${name}=\"${value}\""
            ;;
        *)
            break
            ;;
    esac
    shift
done
