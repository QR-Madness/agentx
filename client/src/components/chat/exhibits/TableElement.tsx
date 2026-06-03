/**
 * TableElement — a structured table that does what a markdown table can't:
 * sort, scroll (sticky header), right-align numeric columns, collapse to stacked
 * cards on mobile, and expand to a full-width modal for large data.
 *
 * Sort state lives here (not in the shared TableView) so it survives expanding.
 */

import { useMemo, useState } from 'react';
import { ChevronDown, ChevronUp, Maximize2 } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../../ui/Dialog';
import { isNumericColumn, sortRows, type TableSort } from './tableSort';
import { memoElement } from './memoElement';
import type { ElementRenderProps } from './types';
import './TableElement.css';

interface TableViewProps {
  columns: string[];
  rows: string[][];
  numericCols: boolean[];
  sort: TableSort | null;
  onSort: (col: number) => void;
}

function TableView({ columns, rows, numericCols, sort, onSort }: TableViewProps) {
  return (
    <div className="exhibit-table-scroll">
      <table className="exhibit-table">
        <thead>
          <tr>
            {columns.map((col, i) => {
              const active = sort?.col === i;
              return (
                <th
                  key={i}
                  aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
                  className={numericCols[i] ? 'is-numeric' : undefined}
                >
                  <button type="button" className="exhibit-table-sort" onClick={() => onSort(i)}>
                    <span>{col}</span>
                    {active &&
                      (sort.dir === 'asc' ? <ChevronUp size={12} /> : <ChevronDown size={12} />)}
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {columns.map((col, c) => (
                <td key={c} data-label={col} className={numericCols[c] ? 'is-numeric' : undefined}>
                  {row[c] ?? ''}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TableElementImpl({ element }: ElementRenderProps) {
  const [sort, setSort] = useState<TableSort | null>(null);

  const columns = element.type === 'table' ? element.columns : [];
  const rawRows = element.type === 'table' ? element.rows : [];
  const caption = element.type === 'table' ? element.caption : undefined;
  const title = element.title;

  const numericCols = useMemo(
    () => columns.map((_, i) => isNumericColumn(rawRows, i)),
    [columns, rawRows],
  );
  const rows = useMemo(
    () => (sort ? sortRows(rawRows, sort.col, sort.dir) : rawRows),
    [rawRows, sort],
  );

  const onSort = (col: number) =>
    setSort((prev) =>
      prev && prev.col === col
        ? { col, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
        : { col, dir: 'asc' },
    );

  if (element.type !== 'table') return null;

  const view = (
    <TableView columns={columns} rows={rows} numericCols={numericCols} sort={sort} onSort={onSort} />
  );

  return (
    <figure className="m-0 flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2">
        {title ? <figcaption className="text-sm font-medium text-fg">{title}</figcaption> : <span />}
        <Dialog>
          <DialogTrigger asChild>
            <button
              type="button"
              className="inline-flex items-center gap-1 text-xs text-fg-muted hover:text-fg"
            >
              <Maximize2 size={12} /> Expand
            </button>
          </DialogTrigger>
          <DialogContent className="max-w-[90vw]">
            <DialogHeader>
              <DialogTitle>{title || caption || 'Table'}</DialogTitle>
            </DialogHeader>
            <div className="p-6 pt-2">{view}</div>
          </DialogContent>
        </Dialog>
      </div>
      {view}
      {caption && <figcaption className="text-xs text-fg-muted">{caption}</figcaption>}
    </figure>
  );
}

export const TableElement = memoElement(TableElementImpl);
