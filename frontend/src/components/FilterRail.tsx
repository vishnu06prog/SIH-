import { useState } from "react";
import type { Facets, FilterOptions, SearchQuery } from "../lib/types";

interface Props {
  options: FilterOptions | null;
  facets: Facets | null;
  query: SearchQuery;
  onToggle: (key: FilterKey, value: string) => void;
  onDateRange: (value: string) => void;
  onReset: () => void;
}

export type FilterKey = "departments" | "priorities" | "fileTypes" | "statuses";

const COLLAPSE_AFTER = 6;

function FilterGroup({
  title,
  values,
  selected,
  counts,
  labels,
  onToggle,
}: {
  title: string;
  values: string[];
  selected: string[];
  counts: Record<string, number>;
  labels?: Record<string, string>;
  onToggle: (value: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  // Keep the canonical KMRL order (High → Medium → Low, departments as listed);
  // only push selected values to the front and empty ones to the back.
  const ordered = values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => {
      const selectedDelta = Number(selected.includes(b.value)) - Number(selected.includes(a.value));
      if (selectedDelta !== 0) return selectedDelta;
      const emptyDelta = Number((counts[a.value] ?? 0) === 0) - Number((counts[b.value] ?? 0) === 0);
      if (emptyDelta !== 0) return emptyDelta;
      return a.index - b.index;
    })
    .map((entry) => entry.value);
  const visible = expanded ? ordered : ordered.slice(0, COLLAPSE_AFTER);

  return (
    <div className="filter-group">
      <h4>{title}</h4>
      <div className="chip-row">
        {visible.map((value) => {
          const count = counts[value] ?? 0;
          const isSelected = selected.includes(value);
          return (
            <button
              key={value}
              type="button"
              className={`chip${isSelected ? " selected" : ""}`}
              aria-pressed={isSelected}
              disabled={!isSelected && count === 0}
              style={!isSelected && count === 0 ? { opacity: 0.42, cursor: "not-allowed" } : undefined}
              onClick={() => onToggle(value)}
            >
              {labels?.[value] ?? value}
              <span className="chip-count">{count}</span>
            </button>
          );
        })}
      </div>
      {ordered.length > COLLAPSE_AFTER ? (
        <button type="button" className="filter-more" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Show fewer" : `Show all ${ordered.length}`}
        </button>
      ) : null}
    </div>
  );
}

function toCounts(list: { value: string; count: number }[] | undefined): Record<string, number> {
  const counts: Record<string, number> = {};
  (list ?? []).forEach((row) => {
    counts[row.value] = row.count;
  });
  return counts;
}

export function FilterRail({ options, facets, query, onToggle, onDateRange, onReset }: Props) {
  if (!options) {
    return (
      <aside className="card filter-rail">
        <div className="card-body col gap-sm">
          <div className="skeleton" style={{ height: 14, width: "60%" }} />
          <div className="skeleton" style={{ height: 28 }} />
          <div className="skeleton" style={{ height: 28 }} />
        </div>
      </aside>
    );
  }

  const statusLabels: Record<string, string> = {};
  options.statuses.forEach((status) => {
    statusLabels[status.value] = status.label;
  });

  const hasFilters =
    query.departments.length ||
    query.priorities.length ||
    query.fileTypes.length ||
    query.statuses.length ||
    query.dateRange !== "any";

  return (
    <aside className="card filter-rail">
      <div className="card-head">
        <h3>Filters</h3>
        {hasFilters ? (
          <div className="card-head-actions">
            <button type="button" className="btn btn-ghost btn-sm" onClick={onReset}>
              Reset
            </button>
          </div>
        ) : null}
      </div>
      <div className="card-body col gap-md">
        <FilterGroup
          title="Department"
          values={options.departments}
          selected={query.departments}
          counts={toCounts(facets?.departments)}
          onToggle={(value) => onToggle("departments", value)}
        />
        <FilterGroup
          title="Priority"
          values={options.priorities}
          selected={query.priorities}
          counts={toCounts(facets?.priorities)}
          onToggle={(value) => onToggle("priorities", value)}
        />
        <FilterGroup
          title="File type"
          values={options.file_types}
          selected={query.fileTypes}
          counts={toCounts(facets?.file_types)}
          onToggle={(value) => onToggle("fileTypes", value)}
        />
        <FilterGroup
          title="Processing status"
          values={options.statuses.map((status) => status.value)}
          selected={query.statuses}
          counts={toCounts(facets?.statuses)}
          labels={statusLabels}
          onToggle={(value) => onToggle("statuses", value)}
        />
        <div className="filter-group">
          <h4>Upload date</h4>
          <select
            className="select"
            value={query.dateRange}
            onChange={(event) => onDateRange(event.target.value)}
            aria-label="Filter by upload date"
          >
            {options.date_ranges.map((range) => (
              <option key={range.value} value={range.value}>
                {range.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </aside>
  );
}
