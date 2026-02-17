"""
Ramish Explorer CLI — Read, query, and explore .ramish knowledge graph files.

Usage:
    ramish-explorer stats engine.ramish
    ramish-explorer query engine.ramish "What relates to AC/DC?"
    ramish-explorer explore engine.ramish
"""

import sys
import click
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .reader import RamishFile
from .quate import hamilton_product_np, quaternion_conjugate_np

console = Console()


def load_file(path: str) -> RamishFile:
    """Load a .ramish file with error handling."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found:[/red] {path}")
        sys.exit(1)
    if not p.suffix == '.ramish':
        console.print(f"[yellow]Warning:[/yellow] File does not have .ramish extension")
    try:
        return RamishFile.load(str(p))
    except Exception as e:
        console.print(f"[red]Failed to load:[/red] {e}")
        sys.exit(1)


def weight_color(w: float) -> str:
    """Color code for truth weight."""
    if w >= 0.7:
        return "green"
    elif w >= 0.4:
        return "yellow"
    else:
        return "red"


def weight_bar(w: float, width: int = 20) -> str:
    """Visual bar for truth weight."""
    filled = int(w * width)
    return "█" * filled + "░" * (width - filled)


@click.group()
@click.version_option(version="0.1.0", prog_name="ramish-explorer")
def cli():
    """Ramish Explorer — Explore .ramish knowledge graph files."""
    pass


# ─── stats ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def stats(file):
    """Show file structure, geometry, and truth weight distribution."""
    rf = load_file(file)
    s = rf.get_stats()

    console.print()
    console.print(Panel.fit(
        f"[bold]{Path(file).name}[/bold]",
        title="Ramish Engine File",
        border_style="blue"
    ))

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Key", style="cyan", width=24)
    table.add_column("Value", style="white")

    table.add_row("Entities", f"{s.entity_count:,}")
    table.add_row("Relations", f"{s.relation_count:,}")
    table.add_row("Embedding Dimension", f"{s.embedding_dim}  (quaternion: {s.embedding_dim}d × 4 = {s.embedding_dim * 4} components)")
    table.add_row("File Size", f"{s.file_size_mb:.2f} MB")
    table.add_row("Quantization", rf._quantize_dtype)

    # Entity types
    if rf.entity_types:
        type_counts = {}
        for e in rf.entities:
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
        type_str = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))
        table.add_row("Entity Types", type_str)

    # Relation types
    if rf.relation_type_names:
        rel_str = ", ".join(sorted(rf.relation_type_names.values()))
        table.add_row("Relation Types", rel_str)

    table.add_row("", "")
    table.add_row("Trust Distribution", "")
    table.add_row(f"  High (>0.7)", f"[green]{s.high_confidence_pct:.1f}%[/green]  {weight_bar(s.high_confidence_pct / 100)}")
    table.add_row(f"  Medium (0.3–0.7)", f"[yellow]{s.medium_confidence_pct:.1f}%[/yellow]  {weight_bar(s.medium_confidence_pct / 100)}")
    table.add_row(f"  Low (<0.3)", f"[red]{s.low_confidence_pct:.1f}%[/red]  {weight_bar(s.low_confidence_pct / 100)}")

    if rf.truth_weights is not None and len(rf.truth_weights) > 0:
        table.add_row("", "")
        table.add_row("Mean Trust Weight", f"{float(np.mean(rf.truth_weights)):.4f}")
        table.add_row("Median Trust Weight", f"{float(np.median(rf.truth_weights)):.4f}")
        table.add_row("Std Dev", f"{float(np.std(rf.truth_weights)):.4f}")

    console.print(table)

    # Top hubs
    hubs = rf.get_top_hubs(5)
    if hubs:
        console.print()
        hub_table = Table(title="Top Hubs", box=box.ROUNDED)
        hub_table.add_column("Entity", style="bold")
        hub_table.add_column("Type", style="dim")
        hub_table.add_column("Degree", justify="right")
        hub_table.add_column("Strong", justify="right", style="green")
        hub_table.add_column("Weak", justify="right", style="red")
        hub_table.add_column("Avg Weight", justify="right")

        for h in hubs:
            hub_table.add_row(
                h.name, h.entity_type, str(h.degree),
                str(h.thick_cables), str(h.loose_threads),
                f"{h.avg_weight:.3f}"
            )
        console.print(hub_table)


# ─── query ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('query_text')
@click.option('--topk', '-k', default=10, help='Number of results')
def query(file, query_text, topk):
    """Natural language search with wave propagation."""
    rf = load_file(file)
    results = rf.query(query_text, topk=topk)

    if not results:
        console.print(f"[yellow]No results found for:[/yellow] {query_text}")
        return

    console.print()
    table = Table(title=f"Query: \"{query_text}\"", box=box.ROUNDED)
    table.add_column("Subject", style="bold")
    table.add_column("Relation", style="cyan")
    table.add_column("Object", style="bold")
    table.add_column("Trust", justify="right")
    table.add_column("", width=20)

    for r in results:
        color = weight_color(r.truth_weight)
        table.add_row(
            r.subject, r.relation, r.object,
            f"[{color}]{r.truth_weight:.3f}[/{color}]",
            weight_bar(r.truth_weight)
        )

    console.print(table)
    console.print(f"  [dim]{len(results)} results[/dim]")


# ─── validate ─────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('subject')
@click.argument('relation')
@click.argument('object_name')
def validate(file, subject, relation, object_name):
    """Check a specific triple (subject, relation, object)."""
    rf = load_file(file)
    result = rf.validate_claim(subject, relation, object_name)

    console.print()
    color = weight_color(result.truth_weight)

    panel_text = Text()
    panel_text.append(f"  {subject}", style="bold")
    panel_text.append(f"  —[{relation}]→  ", style="cyan")
    panel_text.append(f"{object_name}\n\n", style="bold")
    panel_text.append(f"  Truth Weight: ", style="dim")
    panel_text.append(f"{result.truth_weight:.4f}\n", style=color)
    panel_text.append(f"  Support Paths: ", style="dim")
    panel_text.append(f"{result.supporting_paths}\n", style="white")
    panel_text.append(f"  Verdict: ", style="dim")
    panel_text.append(f"{result.verdict}", style=color)

    console.print(Panel(panel_text, title="Claim Validation", border_style=color))


# ─── edges ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('entity_name')
@click.option('--limit', '-n', default=50, help='Maximum edges to show')
def edges(file, entity_name, limit):
    """Show all edges for an entity with trust weights."""
    rf = load_file(file)
    relations = rf.get_relations(entity_name)

    if not relations:
        # Try fuzzy match
        matches = [name for name in rf.name_to_id.keys() if entity_name.lower() in name]
        if matches:
            console.print(f"[yellow]Entity '{entity_name}' not found. Did you mean:[/yellow]")
            for m in matches[:10]:
                console.print(f"  • {m}")
        else:
            console.print(f"[red]Entity not found:[/red] {entity_name}")
        return

    console.print()
    table = Table(title=f"Edges: {entity_name}", box=box.ROUNDED)
    table.add_column("Relation", style="cyan")
    table.add_column("Target", style="bold")
    table.add_column("Trust", justify="right")
    table.add_column("", width=20)

    for r in relations[:limit]:
        color = weight_color(r.truth_weight)
        table.add_row(
            r.relation, r.target,
            f"[{color}]{r.truth_weight:.3f}[/{color}]",
            weight_bar(r.truth_weight)
        )

    console.print(table)
    total = len(relations)
    shown = min(total, limit)
    console.print(f"  [dim]Showing {shown} of {total} edges[/dim]")


# ─── similar ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('entity_name')
@click.option('--topk', '-k', default=10, help='Number of neighbors')
def similar(file, entity_name, topk):
    """Find nearest neighbors by embedding distance."""
    rf = load_file(file)

    entity_id = rf.name_to_id.get(entity_name.lower())
    if entity_id is None:
        matches = [name for name in rf.name_to_id.keys() if entity_name.lower() in name]
        if matches:
            console.print(f"[yellow]Entity '{entity_name}' not found. Did you mean:[/yellow]")
            for m in matches[:10]:
                console.print(f"  • {m}")
        else:
            console.print(f"[red]Entity not found:[/red] {entity_name}")
        return

    if rf.embeddings is None:
        console.print("[red]No embeddings in this file[/red]")
        return

    target_emb = rf.embeddings[entity_id]
    # Cosine distance to all other entities
    norms = np.linalg.norm(rf.embeddings, axis=1)
    target_norm = np.linalg.norm(target_emb)

    if target_norm < 1e-8:
        console.print("[red]Entity has zero embedding[/red]")
        return

    similarities = rf.embeddings @ target_emb / (norms * target_norm + 1e-8)
    # Exclude self
    similarities[entity_id] = -1.0

    top_indices = np.argsort(similarities)[::-1][:topk]

    console.print()
    table = Table(title=f"Similar to: {entity_name}", box=box.ROUNDED)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Entity", style="bold")
    table.add_column("Type", style="dim")
    table.add_column("Similarity", justify="right")
    table.add_column("", width=20)

    for rank, idx in enumerate(top_indices, 1):
        entity = rf.id_to_entity.get(idx)
        if not entity:
            continue
        sim = similarities[idx]
        color = "green" if sim > 0.8 else "yellow" if sim > 0.5 else "white"
        table.add_row(
            str(rank), entity.name, entity.entity_type,
            f"[{color}]{sim:.4f}[/{color}]",
            weight_bar(max(0, sim))
        )

    console.print(table)


# ─── predict ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('head_entity')
@click.argument('relation')
@click.option('--topk', '-k', default=10, help='Number of predictions')
def predict(file, head_entity, relation, topk):
    """Frozen key rotation inference: head + relation → predicted tail."""
    rf = load_file(file)

    head_id = rf.name_to_id.get(head_entity.lower())
    if head_id is None:
        console.print(f"[red]Entity not found:[/red] {head_entity}")
        return

    rel_type = rf.relation_types.get(relation)
    if rel_type is None:
        for name, rt in rf.relation_types.items():
            if relation.lower() in name.lower():
                rel_type = rt
                relation = name
                break

    if rel_type is None:
        console.print(f"[red]Relation not found:[/red] {relation}")
        console.print(f"Available relations: {', '.join(rf.relation_types.keys())}")
        return

    if rf.embeddings is None:
        console.print("[red]No embeddings in this file[/red]")
        return

    head_emb = rf.embeddings[head_id]
    dim = rf.embeddings.shape[1] // 4

    # Extract quaternion components for head
    head_q = head_emb.reshape(4, dim).T  # (dim, 4)

    # We need relation embeddings — for now use the structural approach
    # Find all entities connected by this relation to compute a "frozen key"
    rel_targets = []
    for r in rf.relations:
        if r.relation_type == rel_type and r.head_id == head_id:
            rel_targets.append(r.tail_id)

    if rel_targets:
        # Average target embedding as prediction baseline
        target_embs = rf.embeddings[rel_targets]
        predicted = np.mean(target_embs, axis=0)
    else:
        # Use Hamilton product with structural key estimation
        # Find all instances of this relation type to estimate the relational key
        heads = []
        tails = []
        for r in rf.relations:
            if r.relation_type == rel_type:
                heads.append(r.head_id)
                tails.append(r.tail_id)

        if not heads:
            console.print(f"[yellow]No instances of relation '{relation}' found[/yellow]")
            return

        head_embs = rf.embeddings[heads]
        tail_embs = rf.embeddings[tails]

        # Frozen key: average of conjugate(head) * tail
        h_q = head_embs.reshape(len(heads), 4, dim).transpose(0, 2, 1)  # (n, dim, 4)
        t_q = tail_embs.reshape(len(tails), 4, dim).transpose(0, 2, 1)  # (n, dim, 4)

        h_conj = quaternion_conjugate_np(h_q)
        keys = hamilton_product_np(h_conj, t_q)
        frozen_key = np.mean(keys, axis=0)  # (dim, 4)

        # Predict: head * frozen_key
        head_q_expanded = head_q[np.newaxis, ...]  # (1, dim, 4)
        fk_expanded = frozen_key[np.newaxis, ...]  # (1, dim, 4)
        predicted_q = hamilton_product_np(head_q_expanded, fk_expanded)
        predicted = predicted_q.squeeze().T.flatten()  # back to (dim*4,)

    # Find nearest entities to prediction
    distances = np.linalg.norm(rf.embeddings - predicted, axis=1)
    top_indices = np.argsort(distances)[:topk + 1]

    console.print()
    table = Table(title=f"Predict: {head_entity} —[{relation}]→ ?", box=box.ROUNDED)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Predicted Entity", style="bold")
    table.add_column("Type", style="dim")
    table.add_column("Distance", justify="right")

    rank = 1
    for idx in top_indices:
        if idx == head_id:
            continue
        entity = rf.id_to_entity.get(idx)
        if not entity:
            continue
        dist = distances[idx]
        color = "green" if dist < 1.0 else "yellow" if dist < 3.0 else "white"
        table.add_row(
            str(rank), entity.name, entity.entity_type,
            f"[{color}]{dist:.4f}[/{color}]"
        )
        rank += 1
        if rank > topk:
            break

    console.print(table)


# ─── compose ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('relation1')
@click.argument('relation2')
def compose(file, relation1, relation2):
    """Multi-hop quaternion composition: relation1 ⊗ relation2."""
    rf = load_file(file)

    rt1 = rf.relation_types.get(relation1)
    rt2 = rf.relation_types.get(relation2)

    if rt1 is None:
        for name, rt in rf.relation_types.items():
            if relation1.lower() in name.lower():
                rt1 = rt
                relation1 = name
                break
    if rt2 is None:
        for name, rt in rf.relation_types.items():
            if relation2.lower() in name.lower():
                rt2 = rt
                relation2 = name
                break

    if rt1 is None or rt2 is None:
        console.print(f"[red]Relation(s) not found.[/red] Available: {', '.join(rf.relation_types.keys())}")
        return

    if rf.embeddings is None:
        console.print("[red]No embeddings in this file[/red]")
        return

    dim = rf.embeddings.shape[1] // 4

    # Extract frozen keys for each relation
    def extract_key(rel_type):
        heads, tails = [], []
        for r in rf.relations:
            if r.relation_type == rel_type:
                heads.append(r.head_id)
                tails.append(r.tail_id)
        if not heads:
            return None
        h = rf.embeddings[heads].reshape(len(heads), 4, dim).transpose(0, 2, 1)
        t = rf.embeddings[tails].reshape(len(tails), 4, dim).transpose(0, 2, 1)
        keys = hamilton_product_np(quaternion_conjugate_np(h), t)
        return np.mean(keys, axis=0)

    key1 = extract_key(rt1)
    key2 = extract_key(rt2)

    if key1 is None or key2 is None:
        console.print("[red]Could not extract frozen keys (no instances found)[/red]")
        return

    # Compose: key1 ⊗ key2
    composed = hamilton_product_np(
        key1[np.newaxis, ...],
        key2[np.newaxis, ...]
    ).squeeze()

    # Compute angular distance between composed and individual keys
    def quat_norm(q):
        return np.sqrt(np.sum(q ** 2, axis=-1, keepdims=True))

    key1_norm = quat_norm(key1)
    key2_norm = quat_norm(key2)
    composed_norm = quat_norm(composed)

    console.print()
    console.print(Panel.fit(
        f"[cyan]{relation1}[/cyan] ⊗ [cyan]{relation2}[/cyan]\n\n"
        f"Key 1 avg norm: {float(np.mean(key1_norm)):.4f}\n"
        f"Key 2 avg norm: {float(np.mean(key2_norm)):.4f}\n"
        f"Composed avg norm: {float(np.mean(composed_norm)):.4f}",
        title="Quaternion Composition",
        border_style="cyan"
    ))

    console.print(f"\n  [dim]The composed key represents the multi-hop path:[/dim]")
    console.print(f"  [bold]? —[{relation1}]→ ? —[{relation2}]→ ?[/bold]")


# ─── keys ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def keys(file):
    """Extract frozen relational keys with stability metrics."""
    rf = load_file(file)

    if rf.embeddings is None:
        console.print("[red]No embeddings in this file[/red]")
        return

    dim = rf.embeddings.shape[1] // 4

    console.print()
    table = Table(title="Frozen Relational Keys", box=box.ROUNDED)
    table.add_column("Relation", style="cyan")
    table.add_column("Instances", justify="right")
    table.add_column("Avg Norm", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Stability", justify="right")

    for rel_name, rel_type in sorted(rf.relation_types.items(), key=lambda x: x[0]):
        heads, tails = [], []
        for r in rf.relations:
            if r.relation_type == rel_type:
                heads.append(r.head_id)
                tails.append(r.tail_id)

        if not heads:
            continue

        h = rf.embeddings[heads].reshape(len(heads), 4, dim).transpose(0, 2, 1)
        t = rf.embeddings[tails].reshape(len(tails), 4, dim).transpose(0, 2, 1)
        individual_keys = hamilton_product_np(quaternion_conjugate_np(h), t)
        mean_key = np.mean(individual_keys, axis=0)

        # Stability: how consistent are individual keys vs the mean?
        deviations = np.linalg.norm(individual_keys - mean_key, axis=-1)
        avg_dev = float(np.mean(deviations))
        key_norm = float(np.mean(np.linalg.norm(mean_key, axis=-1)))

        stability = max(0, 1.0 - avg_dev / max(key_norm, 1e-8))
        color = "green" if stability > 0.8 else "yellow" if stability > 0.5 else "red"

        table.add_row(
            rel_name, str(len(heads)),
            f"{key_norm:.4f}", f"{avg_dev:.4f}",
            f"[{color}]{stability:.3f}[/{color}]"
        )

    console.print(table)


# ─── audit ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def audit(file):
    """Structural integrity report from .ramish file."""
    rf = load_file(file)
    result = rf.audit()

    console.print()

    score_color = "green" if result.overall_score > 0.8 else "yellow" if result.overall_score > 0.5 else "red"
    console.print(Panel.fit(
        f"[{score_color}]Overall Score: {result.overall_score:.2f}[/{score_color}]",
        title="Data Quality Audit",
        border_style=score_color
    ))

    if result.issues:
        console.print()
        table = Table(title="Issues Found", box=box.ROUNDED)
        table.add_column("Severity", width=10)
        table.add_column("Description")
        table.add_column("Affected", justify="right")

        severity_colors = {"high": "red", "medium": "yellow", "low": "dim"}
        for issue in result.issues:
            color = severity_colors.get(issue.severity, "white")
            table.add_row(
                f"[{color}]{issue.severity.upper()}[/{color}]",
                issue.description,
                str(issue.affected_count)
            )
        console.print(table)

    if result.recommendations:
        console.print()
        console.print("[bold]Recommendations:[/bold]")
        for rec in result.recommendations:
            console.print(f"  → {rec}")

    if not result.issues:
        console.print(f"\n  [green]No structural issues detected.[/green]")


# ─── requantize ───────────────────────────────────────────────────────────────

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--dtype', type=click.Choice(['fp32', 'fp16', 'int8']), required=True,
              help='Target quantization')
def requantize(input_file, output_file, dtype):
    """Convert between fp32/fp16/int8 quantization."""
    rf = load_file(input_file)
    old_dtype = rf._quantize_dtype

    rf.save(Path(output_file), quantize=dtype)

    old_size = Path(input_file).stat().st_size
    new_size = Path(output_file).stat().st_size
    ratio = old_size / max(new_size, 1)

    console.print(f"\n  [green]Requantized:[/green] {old_dtype} → {dtype}")
    console.print(f"  {old_size:,} bytes → {new_size:,} bytes ({ratio:.1f}x)")


# ─── explore ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def explore(file):
    """Interactive REPL for exploring .ramish files."""
    rf = load_file(file)

    console.print()
    console.print(Panel.fit(
        f"[bold]Ramish Explorer[/bold] — interactive mode\n"
        f"File: {Path(file).name}\n"
        f"Entities: {len(rf.entities):,}  |  Relations: {len(rf.relations):,}\n\n"
        f"Commands:\n"
        f"  [cyan]search[/cyan] <text>        — natural language query\n"
        f"  [cyan]edges[/cyan] <entity>       — show connections\n"
        f"  [cyan]similar[/cyan] <entity>     — nearest neighbors\n"
        f"  [cyan]predict[/cyan] <entity> <rel> — link prediction\n"
        f"  [cyan]validate[/cyan] <s> <r> <o>  — check a triple\n"
        f"  [cyan]compose[/cyan] <r1> <r2>     — multi-hop composition\n"
        f"  [cyan]keys[/cyan]                 — frozen relational keys\n"
        f"  [cyan]audit[/cyan]                — structural integrity\n"
        f"  [cyan]types[/cyan]                — list entity types\n"
        f"  [cyan]relations[/cyan]            — list relation types\n"
        f"  [cyan]list[/cyan] [type]          — list entities\n"
        f"  [cyan]hubs[/cyan]                 — top connected entities\n"
        f"  [cyan]stats[/cyan]                — file statistics\n"
        f"  [cyan]help[/cyan]                 — show this message\n"
        f"  [cyan]quit[/cyan]                 — exit",
        border_style="blue"
    ))

    while True:
        try:
            user_input = console.input("\n[bold blue]ramish>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        elif cmd in ("help", "h", "?"):
            console.print(
                "  [cyan]search[/cyan] <text>        — natural language query\n"
                "  [cyan]edges[/cyan] <entity>       — show connections\n"
                "  [cyan]similar[/cyan] <entity>     — nearest neighbors\n"
                "  [cyan]predict[/cyan] <entity> <rel> — link prediction\n"
                "  [cyan]validate[/cyan] <s> <r> <o>  — check a triple\n"
                "  [cyan]compose[/cyan] <r1> <r2>     — multi-hop composition\n"
                "  [cyan]keys[/cyan]                 — frozen relational keys\n"
                "  [cyan]audit[/cyan]                — structural integrity\n"
                "  [cyan]types[/cyan]                — list entity types\n"
                "  [cyan]relations[/cyan]            — list relation types\n"
                "  [cyan]list[/cyan] [type]          — list entities\n"
                "  [cyan]hubs[/cyan]                 — top connected entities\n"
                "  [cyan]stats[/cyan]                — file statistics\n"
                "  [cyan]quit[/cyan]                 — exit"
            )

        elif cmd == "search":
            if not arg:
                console.print("[yellow]Usage: search <query text>[/yellow]")
                continue
            results = rf.query(arg, topk=10)
            if not results:
                console.print(f"[yellow]No results for: {arg}[/yellow]")
            else:
                for r in results:
                    color = weight_color(r.truth_weight)
                    console.print(
                        f"  [{color}]{r.truth_weight:.3f}[/{color}]  "
                        f"{r.subject} —[{r.relation}]→ {r.object}"
                    )

        elif cmd == "edges":
            if not arg:
                console.print("[yellow]Usage: edges <entity name>[/yellow]")
                continue
            relations = rf.get_relations(arg)
            if not relations:
                matches = [n for n in rf.name_to_id.keys() if arg.lower() in n]
                if matches:
                    console.print(f"[yellow]Not found. Did you mean:[/yellow]")
                    for m in matches[:8]:
                        console.print(f"  • {m}")
                else:
                    console.print(f"[red]Entity not found: {arg}[/red]")
            else:
                for r in relations[:30]:
                    color = weight_color(r.truth_weight)
                    console.print(
                        f"  [{color}]{r.truth_weight:.3f}[/{color}]  "
                        f"—[{r.relation}]→ {r.target}"
                    )
                if len(relations) > 30:
                    console.print(f"  [dim]... and {len(relations) - 30} more[/dim]")

        elif cmd == "similar":
            if not arg:
                console.print("[yellow]Usage: similar <entity name>[/yellow]")
                continue
            eid = rf.name_to_id.get(arg.lower())
            if eid is None:
                console.print(f"[red]Entity not found: {arg}[/red]")
                continue
            if rf.embeddings is None:
                console.print("[red]No embeddings[/red]")
                continue
            target = rf.embeddings[eid]
            norms = np.linalg.norm(rf.embeddings, axis=1)
            tn = np.linalg.norm(target)
            sims = rf.embeddings @ target / (norms * tn + 1e-8)
            sims[eid] = -1
            top = np.argsort(sims)[::-1][:10]
            for rank, idx in enumerate(top, 1):
                e = rf.id_to_entity.get(idx)
                if e:
                    console.print(f"  {rank:2d}. {sims[idx]:.4f}  {e.name} ({e.entity_type})")

        elif cmd == "validate":
            parts = arg.split()
            if len(parts) < 3:
                console.print("[yellow]Usage: validate <subject> <relation> <object>[/yellow]")
                continue
            result = rf.validate_claim(parts[0], parts[1], " ".join(parts[2:]))
            color = weight_color(result.truth_weight)
            console.print(f"  [{color}]Weight: {result.truth_weight:.4f}[/{color}]")
            console.print(f"  {result.verdict}")

        elif cmd == "types":
            type_counts = {}
            for e in rf.entities:
                type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
                console.print(f"  {t}: {c:,}")

        elif cmd == "list":
            entities = rf.list_entities(type_filter=arg if arg else None, limit=20)
            for e in entities:
                console.print(f"  {e.name} ({e.entity_type})")
            if len(entities) == 20:
                console.print(f"  [dim]... showing first 20[/dim]")

        elif cmd == "hubs":
            hubs = rf.get_top_hubs(10)
            for h in hubs:
                console.print(
                    f"  {h.degree:4d} edges  [{weight_color(h.avg_weight)}]{h.avg_weight:.3f}[/{weight_color(h.avg_weight)}]  "
                    f"{h.name} ({h.entity_type})"
                )

        elif cmd == "stats":
            s = rf.get_stats()
            console.print(f"  Entities: {s.entity_count:,}")
            console.print(f"  Relations: {s.relation_count:,}")
            console.print(f"  Embedding dim: {s.embedding_dim}")
            console.print(f"  File size: {s.file_size_mb:.2f} MB")
            console.print(f"  Trust: {s.high_confidence_pct:.1f}% high, {s.medium_confidence_pct:.1f}% medium, {s.low_confidence_pct:.1f}% low")

        elif cmd == "predict":
            parts_p = arg.split()
            if len(parts_p) < 2:
                console.print("[yellow]Usage: predict <entity> <relation>[/yellow]")
                console.print(f"  [dim]Relations: {', '.join(rf.relation_types.keys())}[/dim]")
                continue
            rel_name = parts_p[-1]
            ent_name = " ".join(parts_p[:-1])
            head_id = rf.name_to_id.get(ent_name.lower())
            if head_id is None:
                console.print(f"[red]Entity not found: {ent_name}[/red]")
                continue
            rel_type = rf.relation_types.get(rel_name)
            if rel_type is None:
                for rn, rt in rf.relation_types.items():
                    if rel_name.lower() in rn.lower():
                        rel_type = rt
                        rel_name = rn
                        break
            if rel_type is None:
                console.print(f"[red]Relation not found: {rel_name}[/red]")
                console.print(f"  [dim]Available: {', '.join(rf.relation_types.keys())}[/dim]")
                continue
            if rf.embeddings is None:
                console.print("[red]No embeddings[/red]")
                continue
            dim = rf.embeddings.shape[1] // 4
            heads_p, tails_p = [], []
            for r in rf.relations:
                if r.relation_type == rel_type:
                    heads_p.append(r.head_id)
                    tails_p.append(r.tail_id)
            if not heads_p:
                console.print(f"[yellow]No instances of '{rel_name}' found[/yellow]")
                continue
            h_q = rf.embeddings[heads_p].reshape(len(heads_p), 4, dim).transpose(0, 2, 1)
            t_q = rf.embeddings[tails_p].reshape(len(tails_p), 4, dim).transpose(0, 2, 1)
            frozen_key = np.mean(hamilton_product_np(quaternion_conjugate_np(h_q), t_q), axis=0)
            head_q = rf.embeddings[head_id].reshape(4, dim).T[np.newaxis, ...]
            predicted_q = hamilton_product_np(head_q, frozen_key[np.newaxis, ...])
            predicted = predicted_q.squeeze().T.flatten()
            distances = np.linalg.norm(rf.embeddings - predicted, axis=1)
            top_idx = np.argsort(distances)[:11]
            console.print(f"  [bold]{ent_name}[/bold] —[[cyan]{rel_name}[/cyan]]→ ?")
            rank = 1
            for idx in top_idx:
                if idx == head_id:
                    continue
                e = rf.id_to_entity.get(idx)
                if not e:
                    continue
                d = distances[idx]
                color = "green" if d < 1.0 else "yellow" if d < 3.0 else "white"
                console.print(f"  {rank:2d}. [{color}]{d:.4f}[/{color}]  {e.name} ({e.entity_type})")
                rank += 1
                if rank > 10:
                    break

        elif cmd == "compose":
            parts_c = arg.split()
            if len(parts_c) < 2:
                console.print("[yellow]Usage: compose <relation1> <relation2>[/yellow]")
                console.print(f"  [dim]Relations: {', '.join(rf.relation_types.keys())}[/dim]")
                continue
            r1_name, r2_name = parts_c[0], parts_c[1]
            rt1, rt2 = rf.relation_types.get(r1_name), rf.relation_types.get(r2_name)
            if rt1 is None:
                for rn, rt in rf.relation_types.items():
                    if r1_name.lower() in rn.lower():
                        rt1 = rt
                        r1_name = rn
                        break
            if rt2 is None:
                for rn, rt in rf.relation_types.items():
                    if r2_name.lower() in rn.lower():
                        rt2 = rt
                        r2_name = rn
                        break
            if rt1 is None or rt2 is None:
                console.print(f"[red]Relation(s) not found.[/red] Available: {', '.join(rf.relation_types.keys())}")
                continue
            if rf.embeddings is None:
                console.print("[red]No embeddings[/red]")
                continue
            dim = rf.embeddings.shape[1] // 4
            def _extract_key(rel_t):
                hh, tt = [], []
                for r in rf.relations:
                    if r.relation_type == rel_t:
                        hh.append(r.head_id)
                        tt.append(r.tail_id)
                if not hh:
                    return None
                h_ = rf.embeddings[hh].reshape(len(hh), 4, dim).transpose(0, 2, 1)
                t_ = rf.embeddings[tt].reshape(len(tt), 4, dim).transpose(0, 2, 1)
                return np.mean(hamilton_product_np(quaternion_conjugate_np(h_), t_), axis=0)
            k1, k2 = _extract_key(rt1), _extract_key(rt2)
            if k1 is None or k2 is None:
                console.print("[red]Could not extract frozen keys[/red]")
                continue
            composed = hamilton_product_np(k1[np.newaxis, ...], k2[np.newaxis, ...]).squeeze()
            k1_norm = float(np.mean(np.linalg.norm(k1, axis=-1)))
            k2_norm = float(np.mean(np.linalg.norm(k2, axis=-1)))
            c_norm = float(np.mean(np.linalg.norm(composed, axis=-1)))
            console.print(f"  [cyan]{r1_name}[/cyan] ⊗ [cyan]{r2_name}[/cyan]")
            console.print(f"  Key 1 norm: {k1_norm:.4f}  |  Key 2 norm: {k2_norm:.4f}  |  Composed: {c_norm:.4f}")
            console.print(f"  [dim]Path: ? —[{r1_name}]→ ? —[{r2_name}]→ ?[/dim]")

        elif cmd == "keys":
            if rf.embeddings is None:
                console.print("[red]No embeddings[/red]")
                continue
            dim = rf.embeddings.shape[1] // 4
            for rn, rt in sorted(rf.relation_types.items()):
                hh, tt = [], []
                for r in rf.relations:
                    if r.relation_type == rt:
                        hh.append(r.head_id)
                        tt.append(r.tail_id)
                if not hh:
                    continue
                h_ = rf.embeddings[hh].reshape(len(hh), 4, dim).transpose(0, 2, 1)
                t_ = rf.embeddings[tt].reshape(len(tt), 4, dim).transpose(0, 2, 1)
                ind_keys = hamilton_product_np(quaternion_conjugate_np(h_), t_)
                mean_key = np.mean(ind_keys, axis=0)
                dev = float(np.mean(np.linalg.norm(ind_keys - mean_key, axis=-1)))
                knorm = float(np.mean(np.linalg.norm(mean_key, axis=-1)))
                stability = max(0, 1.0 - dev / max(knorm, 1e-8))
                color = "green" if stability > 0.8 else "yellow" if stability > 0.5 else "red"
                console.print(f"  [{color}]{stability:.3f}[/{color}]  {rn} ({len(hh)} instances, norm {knorm:.4f})")

        elif cmd == "audit":
            result = rf.audit()
            score_color = "green" if result.overall_score > 0.8 else "yellow" if result.overall_score > 0.5 else "red"
            console.print(f"  [{score_color}]Score: {result.overall_score:.2f}[/{score_color}]")
            if result.issues:
                for issue in result.issues:
                    sev_color = {"high": "red", "medium": "yellow", "low": "dim"}.get(issue.severity, "white")
                    console.print(f"  [{sev_color}]{issue.severity.upper()}[/{sev_color}]  {issue.description}")
            else:
                console.print(f"  [green]No structural issues detected.[/green]")
            for rec in result.recommendations:
                console.print(f"  → {rec}")

        elif cmd == "relations":
            if rf.relation_type_names:
                for rt_id, rt_name in sorted(rf.relation_type_names.items()):
                    count = sum(1 for r in rf.relations if r.relation_type == rt_id)
                    console.print(f"  {rt_name}: {count:,}")
            else:
                console.print("[yellow]No relation type metadata[/yellow]")

        else:
            # Default: treat as search
            results = rf.query(user_input, topk=10)
            if results:
                for r in results:
                    color = weight_color(r.truth_weight)
                    console.print(
                        f"  [{color}]{r.truth_weight:.3f}[/{color}]  "
                        f"{r.subject} —[{r.relation}]→ {r.object}"
                    )
            else:
                console.print(f"[yellow]Unknown command. Type 'help' for commands.[/yellow]")


if __name__ == "__main__":
    cli()
