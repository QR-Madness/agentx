/** The small tier marker shown beside an agent's name across the Organization surface. */

export type ChipTier = 'ambassador' | 'manager' | 'lead' | 'member';

const LABEL: Record<ChipTier, string> = {
  ambassador: 'APEX',
  manager: 'MGR',
  lead: 'LEAD',
  member: 'MEMBER',
};

const TITLE: Record<ChipTier, string> = {
  ambassador: 'Ambassador — above the chain of command',
  manager: 'Manager — owns teams and delegates to their leads',
  lead: 'Lead — works a team and delegates to its members',
  member: 'Member of a team',
};

export function TierChip({ tier }: { tier: ChipTier }) {
  return (
    <span className={`org-tier org-tier--${tier}`} title={TITLE[tier]}>
      {LABEL[tier]}
    </span>
  );
}
