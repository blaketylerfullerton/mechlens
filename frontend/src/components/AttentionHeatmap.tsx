import { useMemo, useState } from "react";
import { scaleSequential } from "d3-scale";
import { interpolateViridis } from "d3-scale-chromatic";

interface AttentionHeatmapProps {
  tokens: string[];
  attention: number[][][][]; // [layer][head][query][key]
}

const CELL_SIZE = 22;

export function AttentionHeatmap({ tokens, attention }: AttentionHeatmapProps) {
  const numLayers = attention.length;
  const numHeads = attention[0]?.length ?? 0;
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [hovered, setHovered] = useState<{ query: number; key: number } | null>(null);

  const matrix = attention[layer]?.[head] ?? [];
  const color = useMemo(() => scaleSequential(interpolateViridis).domain([0, 1]), []);
  const n = tokens.length;
  const size = n * CELL_SIZE;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          Layer
          <select
            className="h-8 rounded-lg border border-input bg-transparent px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
            value={layer}
            onChange={(e) => setLayer(Number(e.target.value))}
          >
            {Array.from({ length: numLayers }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          Head
          <select
            className="h-8 rounded-lg border border-input bg-transparent px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
            value={head}
            onChange={(e) => setHead(Number(e.target.value))}
          >
            {Array.from({ length: numHeads }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
        {hovered && (
          <p className="font-mono text-sm text-muted-foreground">
            <span className="text-foreground">{tokens[hovered.query]}</span> attends to{" "}
            <span className="text-foreground">{tokens[hovered.key]}</span> ·{" "}
            {(matrix[hovered.query][hovered.key] * 100).toFixed(1)}%
          </p>
        )}
      </div>

      <div className="overflow-auto rounded-lg border border-border">
        <svg
          width={size}
          height={size}
          role="img"
          aria-label={`Attention weights for layer ${layer}, head ${head}`}
        >
          {matrix.map((row, q) =>
            row.map((weight, k) => (
              <rect
                key={`${q}-${k}`}
                x={k * CELL_SIZE}
                y={q * CELL_SIZE}
                width={CELL_SIZE}
                height={CELL_SIZE}
                fill={color(weight)}
                stroke={
                  hovered?.query === q && hovered?.key === k
                    ? "var(--color-foreground)"
                    : "transparent"
                }
                strokeWidth={1.5}
                onMouseEnter={() => setHovered({ query: q, key: k })}
                onMouseLeave={() => setHovered(null)}
              >
                <title>
                  {tokens[q]} → {tokens[k]}: {(weight * 100).toFixed(1)}%
                </title>
              </rect>
            ))
          )}
        </svg>
      </div>
      <p className="text-xs text-muted-foreground">
        Rows are query tokens (the token doing the attending), columns are key tokens (the token
        being attended to).
      </p>
    </div>
  );
}
