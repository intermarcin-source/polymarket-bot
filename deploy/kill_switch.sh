#!/bin/bash
# ==============================================
# Polymarket Bot - Emergency Kill Switch
# ==============================================
# Usage:
#   ./kill_switch.sh on "reason"   - HALT all trading immediately
#   ./kill_switch.sh off           - Resume trading
#   ./kill_switch.sh               - Check status
# ==============================================

KILL_FILE="/opt/polymarket-bot/data/KILL_SWITCH"

case "$1" in
    on)
        REASON="${2:-Manual emergency stop}"
        echo "KILL SWITCH ACTIVATED: ${REASON} | $(date -u)" > "$KILL_FILE"
        echo ""
        echo "  KILL SWITCH: ON"
        echo "  Reason: ${REASON}"
        echo "  Bot will stop all trading on next cycle check."
        echo ""
        echo "  To resume: ./kill_switch.sh off"
        ;;
    off)
        rm -f "$KILL_FILE"
        echo ""
        echo "  KILL SWITCH: OFF"
        echo "  Bot will resume trading on next cycle."
        ;;
    *)
        if [ -f "$KILL_FILE" ]; then
            echo ""
            echo "  KILL SWITCH: ACTIVE"
            echo "  Details: $(cat "$KILL_FILE")"
            echo ""
            echo "  To resume: ./kill_switch.sh off"
        else
            echo ""
            echo "  KILL SWITCH: INACTIVE"
            echo "  Bot is trading normally."
            echo ""
            echo "  To halt: ./kill_switch.sh on \"reason\""
        fi
        ;;
esac
