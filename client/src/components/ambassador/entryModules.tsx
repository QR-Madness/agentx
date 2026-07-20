/**
 * entryModules — the Ambassador thread's extensibility seam.
 *
 * Every Inquiry entry (Q&A answer or briefing) renders an ordered stack of
 * **modules** above its prose. A module is a typed payload stamped on the entry
 * server-side (today both ride the persisted `toolCalls` chips — the proposal
 * payload is the precedent) plus one renderer registered here. Adding a future
 * surface — aide-delegation progress, exhibit displays, forms, reports — is one
 * `ENTRY_MODULES` entry; the panel itself needs no surgery.
 *
 * Mirrors the repo's registry precedents (`chat/messageRegistry.ts`,
 * `chat/exhibits/elementRegistry.ts`).
 */

import { Fragment, type ReactNode } from 'react';
import { Check, Loader2, X } from 'lucide-react';
import type { AmbassadorToolCall, AmbassadorToolProposal } from '../../lib/api';
import {
  getProposalResolution,
  proposalSentence,
  toolChipLabel,
} from '../../lib/ambassadorTools';
import { Button } from '../ui';

/** The slice of an Inquiry entry a module may render from. Widen deliberately
 *  when a future module needs a new payload carrier — this is the one contract. */
export interface EntrySurface {
  /** Persisted tool-call chips — today's payload carrier (proposals ride here). */
  toolCalls?: AmbassadorToolCall[];
}

/** Interaction context threaded from the panel (stable across entries). */
export interface EntryModuleCtx {
  /** Only the latest entry renders live interactive affordances. */
  isLatest: boolean;
  /** Bump = proposal resolutions changed in localStorage (re-render trigger). */
  resolutionVersion: number;
  onConfirmProposal: (p: AmbassadorToolProposal) => void | Promise<void>;
  onDismissProposal: (p: AmbassadorToolProposal) => void;
}

export interface AmbassadorEntryModule {
  id: string;
  render: (entry: EntrySurface, ctx: EntryModuleCtx) => ReactNode;
}

/** Live chips for the tools the ambassador called while answering — spinner
 *  while running, check when done — so you can see it reading/surveying.
 *  Confirmed-write proposals get a distinct "needs you" warning tone. */
function ToolChips({ calls }: { calls?: AmbassadorToolCall[] }) {
  if (!calls?.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {calls.map((c, i) => (
        <span
          key={`${c.tool}-${i}`}
          className={
            c.proposal
              ? 'inline-flex items-center gap-1 rounded-full bg-warning/15 px-2 py-0.5 text-[11px] text-warning'
              : 'inline-flex items-center gap-1 rounded-full bg-surface-sunken px-2 py-0.5 text-[11px] text-fg-secondary'
          }
        >
          {c.done ? (
            <Check size={11} className={c.proposal ? 'text-warning' : 'text-success'} />
          ) : (
            <Loader2 size={11} className="animate-spin text-accent" />
          )}
          {toolChipLabel(c.tool, c.args)}
          {!c.done && '…'}
        </span>
      ))}
    </div>
  );
}

/** The confirm strip for the belt's confirmed-write proposals (rename / archive /
 *  delete / dispatch). Only the *latest* entry renders live Confirm/Dismiss
 *  buttons — older or already-actioned proposals collapse to a passive status
 *  line. Nothing executes until the person confirms. */
function ProposalStrips({ entry, ctx }: { entry: EntrySurface; ctx: EntryModuleCtx }) {
  void ctx.resolutionVersion; // re-render trigger — resolutions are read from storage
  const proposals = (entry.toolCalls ?? []).filter((c) => c.proposal).map((c) => c.proposal!);
  if (!proposals.length) return null;
  return (
    <div className="flex flex-col gap-1.5">
      {proposals.map((p) => {
        const resolution = getProposalResolution(p.proposal_id);
        if (resolution === 'confirmed') {
          return (
            <span key={p.proposal_id} className="inline-flex items-center gap-1 text-[11px] text-success">
              <Check size={11} /> {p.action === 'rename' ? `Renamed to "${p.title ?? ''}"` :
                p.action === 'archive' ? 'Archived' :
                p.action === 'unarchive' ? 'Restored' :
                p.action === 'dispatch' ? `Dispatched to ${p.agent_name ?? 'the worker'}` : 'Deleted'}
            </span>
          );
        }
        if (resolution === 'dismissed') {
          return (
            <span key={p.proposal_id} className="inline-flex items-center gap-1 text-[11px] text-fg-muted">
              <X size={11} /> Proposal dismissed
            </span>
          );
        }
        if (!ctx.isLatest) {
          return (
            <span key={p.proposal_id} className="inline-flex items-center gap-1 text-[11px] text-fg-muted">
              proposal expired — ask again to redo it
            </span>
          );
        }
        const danger = p.action === 'delete';
        return (
          <div
            key={p.proposal_id}
            className="flex flex-wrap items-center gap-2 rounded-lg border border-warning/40 bg-warning/10 px-2.5 py-1.5"
          >
            <span className="text-xs text-fg">{proposalSentence(p)}</span>
            <div className="ml-auto flex items-center gap-1.5">
              <Button
                size="sm"
                variant={danger ? 'danger' : 'primary'}
                onClick={() => ctx.onConfirmProposal(p)}
              >
                {danger ? 'Delete…' : p.action === 'dispatch' ? 'Dispatch' : 'Confirm'}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => ctx.onDismissProposal(p)}>
                Dismiss
              </Button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** The ordered module stack. Register future modules here — one entry each. */
export const ENTRY_MODULES: AmbassadorEntryModule[] = [
  { id: 'tool-chips', render: (entry) => <ToolChips calls={entry.toolCalls} /> },
  { id: 'proposal-strips', render: (entry, ctx) => <ProposalStrips entry={entry} ctx={ctx} /> },
];

/** Renders every registered module for one entry, in registry order. */
export function EntryModules({ entry, ctx }: { entry: EntrySurface; ctx: EntryModuleCtx }) {
  return (
    <>
      {ENTRY_MODULES.map((m) => (
        <Fragment key={m.id}>{m.render(entry, ctx)}</Fragment>
      ))}
    </>
  );
}
