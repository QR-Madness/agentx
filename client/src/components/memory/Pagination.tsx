// ─── Pagination ────────────────────────────────────────────────────────────

import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '../ui';

export function Pagination({
  page, hasNext, onPageChange,
}: {
  page: number;
  hasNext: boolean;
  onPageChange: (p: number) => void;
}) {
  return (
    <div className="pagination">
      <Button variant="ghost" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
        <ChevronLeft size={16} />Previous
      </Button>
      <span className="page-info">Page {page}</span>
      <Button variant="ghost" disabled={!hasNext} onClick={() => onPageChange(page + 1)}>
        Next<ChevronRight size={16} />
      </Button>
    </div>
  );
}
