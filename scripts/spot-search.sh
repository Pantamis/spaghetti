#!/usr/bin/env bash
# Runs a split-key search on a (spot) CUDA machine so that it survives being
# reclaimed: the search keeps its checkpoint on disk (and, with S3=..., in S3
# too), picks it up again when restarted, and appends every match to a file.
#
#   BASE=02…(33-byte hex base scan pubkey) PATTERN=sp1qqgmlnmarkets \
#   S3=s3://my-bucket/spaghetti scripts/spot-search.sh
#
#   BASE=02… PATTERN=sprt1qqgmlnmarkets NETWORK=regtest scripts/spot-search.sh
#
# Run it at boot (systemd unit in the README) or by hand in tmux. The first run
# on a GPU model tunes the engine with `spaghetti bench-gpu` (a few minutes)
# and keeps the parameters, per GPU model, next to the checkpoint.
#
# Environment:
#   BASE     base scan pubkey D (or XPUB=xpub6… of m/352'/0'/0'/1')
#   PATTERN  the vanity pattern(s), space separated   [sp1qqgmlnmarkets]
#   NETWORK  mainnet | testnet | signet | regtest, for the tuning and the
#            search alike                             [mainnet]
#   DIR      state directory (checkpoint, params, matches) [~/spaghetti-run]
#   S3       optional s3:// prefix the state is mirrored to every minute
#   BIN      the spaghetti binary built with --features cuda [spaghetti]
#   EXTRA    extra search arguments, e.g. "-k 3" (not the network: NETWORK)
set -euo pipefail

PATTERN=${PATTERN:-sp1qqgmlnmarkets}
DIR=${DIR:-$HOME/spaghetti-run}
BIN=${BIN:-spaghetti}
S3=${S3:-}
NETWORK=${NETWORK:-mainnet}
EXTRA=${EXTRA:-}
if [[ -n ${XPUB:-} ]]; then
    KEY=(--xpub "$XPUB")
elif [[ -n ${BASE:-} ]]; then
    KEY=(-b "$BASE")
else
    echo "set BASE (hex base scan pubkey) or XPUB" >&2
    exit 2
fi

mkdir -p "$DIR"
CKPT=$DIR/search.checkpoint
FOUND=$DIR/found.txt
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -c 'A-Za-z0-9\n' '_')
PARAMS=$DIR/gpu-params-$GPU_MODEL.txt

s3_get() { [[ -n $S3 && ! -f $2 ]] && aws s3 cp --only-show-errors "$S3/$1" "$2" 2>/dev/null || true; }
s3_put() { [[ -n $S3 && -f $1 ]] && aws s3 cp --only-show-errors "$1" "$S3/$(basename "$1")" || true; }

s3_get search.checkpoint "$CKPT"
s3_get found.txt "$FOUND"
s3_get "$(basename "$PARAMS")" "$PARAMS"

# shellcheck disable=SC2086 # PATTERN and EXTRA are word lists
if [[ ! -f $PARAMS ]]; then
    echo "tuning for $GPU_MODEL (once per GPU model)…" >&2
    "$BIN" bench-gpu -n "$NETWORK" --output "$PARAMS" $PATTERN
    s3_put "$PARAMS"
fi

# shellcheck disable=SC2086
"$BIN" -n "$NETWORK" --gpu-params "$PARAMS" --checkpoint "$CKPT" "${KEY[@]}" $EXTRA $PATTERN \
    >>"$FOUND" &
SEARCH=$!

# Mirror the state to S3 every minute.
if [[ -n $S3 ]]; then
    (while sleep 60; do s3_put "$CKPT"; s3_put "$FOUND"; done) &
    SYNC=$!
fi

# Spot interruption notice (IMDSv2): two minutes before the reclaim, stop the
# search so it writes its final checkpoint. The search also stops cleanly on
# the SIGTERM of a normal shutdown.
(
    while kill -0 "$SEARCH" 2>/dev/null; do
        token=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token \
            -H 'X-aws-ec2-metadata-token-ttl-seconds: 300' || true)
        if [[ -n $token ]] && curl -sf -m 2 -H "X-aws-ec2-metadata-token: $token" \
            http://169.254.169.254/latest/meta-data/spot/instance-action >/dev/null; then
            echo "spot interruption notice: stopping the search" >&2
            kill -TERM "$SEARCH" 2>/dev/null || true
            break
        fi
        sleep 5
    done
) &
WATCH=$!

trap 'kill -TERM "$SEARCH" 2>/dev/null || true' INT TERM
# `wait` returns early when a trapped signal arrives: wait again until the
# search itself has exited (and saved its checkpoint).
while :; do
    status=0
    wait "$SEARCH" || status=$?
    kill -0 "$SEARCH" 2>/dev/null || break
done
kill "$WATCH" ${SYNC:+"$SYNC"} 2>/dev/null || true
s3_put "$CKPT"
s3_put "$FOUND"
if [[ $status -eq 0 ]]; then
    echo "search complete, matches in $FOUND:" >&2
    cat "$FOUND" >&2
fi
exit "$status"
