// ─── Pagination ────────────────────────────────────────────────────────────

import { ChevronLeft, ChevronRight } from 'lucide-react';

export function Pagination({
  page, hasNext, onPageChange,
}: {
  page: number;
  hasNext: boolean;
  onPageChange: (p: number) => void;
}) {
  return (
    <div className="pagination">
      <button className="button-ghost" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
        <ChevronLeft size={16} />Previous
      </button>
      <span className="page-info">Page {page}</span>
      <button className="button-ghost" disabled={!hasNext} onClick={() => onPageChange(page + 1)}>
        Next<ChevronRight size={16} />
      </button>
    </div>
  );
}
