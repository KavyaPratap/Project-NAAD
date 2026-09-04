# Owner: S3
# Module: Live telemetry dashboard
# Run in a SEPARATE terminal alongside receiver.py:
#   Terminal 1: python receiver.py
#   Terminal 2: python dashboard.py
# Last validated: [date, commit hash]

from rich.console import Console
from rich.table   import Table
from rich.live    import Live
from rich.panel   import Panel
from rich.columns import Columns
from rich import box
import time
import threading

console = Console()

# Shared metrics dict — updated by receiver.py (or you can read from a file/socket)
_metrics = {
    "ram_free_kb":      0,
    "ram_min_kb":       0,
    "largest_block_kb": 0,
    "vad_active":       False,
    "kws_score":        0.0,
    "triggers":         0,
    "packets_rx":       0,
    "last_transcript":  "(waiting...)",
    "asr_latency_ms":   0,
    "e2e_latency_ms":   0,
    "drop_count":       0,
    "uptime_s":         0,
}
_lock = threading.Lock()
_start_time = time.time()


def update_metric(key: str, value) -> None:
    """Call this from receiver.py to update dashboard values."""
    with _lock:
        _metrics[key] = value


def _build_table() -> Table:
    with _lock:
        uptime = int(time.time() - _start_time)
        m = dict(_metrics)

    table = Table(box=box.ROUNDED, expand=True, show_header=False)
    table.add_column("Metric", style="cyan bold", min_width=22)
    table.add_column("Value",  style="white", min_width=18)

    table.add_row("⏱ Uptime",            f"{uptime}s")
    table.add_row("─" * 22,              "─" * 18)
    table.add_row("🔊 VAD active",        "✅ SPEECH" if m["vad_active"] else "⬜ silence")
    table.add_row("🎯 KWS score",         f"{m['kws_score']:.3f}  (thresh 0.85)")
    table.add_row("🔔 Total triggers",    str(m["triggers"]))
    table.add_row("📦 Packets received",  str(m["packets_rx"]))
    table.add_row("⚠️  Drop count",        str(m["drop_count"]))
    table.add_row("─" * 22,              "─" * 18)
    table.add_row("🧠 RAM free",          f"{m['ram_free_kb']} KB")
    table.add_row("📉 RAM min ever",      f"{m['ram_min_kb']} KB")
    table.add_row("📦 Largest block",     f"{m['largest_block_kb']} KB")
    table.add_row("─" * 22,              "─" * 18)
    table.add_row("⏱ ASR latency",       f"{m['asr_latency_ms']} ms")
    table.add_row("⏱ E2E latency",       f"{m['e2e_latency_ms']} ms")

    return table


def _build_transcript_panel() -> Panel:
    with _lock:
        transcript = _metrics["last_transcript"]
    return Panel(
        f"[bold green]{transcript}[/bold green]",
        title="📝 Last Transcript",
        border_style="green"
    )


def run_dashboard() -> None:
    console.print(Panel.fit(
        "[bold cyan]Project NAAD — Live Telemetry Dashboard[/bold cyan]\n"
        "[dim]Press Ctrl+C to stop[/dim]",
        border_style="cyan"
    ))

    with Live(refresh_per_second=2, console=console) as live:
        try:
            while True:
                layout = Columns([
                    Panel(_build_table(), title="📊 System Metrics", border_style="blue"),
                    _build_transcript_panel()
                ])
                live.update(layout)
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("[yellow]Dashboard stopped.[/yellow]")


if __name__ == "__main__":
    run_dashboard()
