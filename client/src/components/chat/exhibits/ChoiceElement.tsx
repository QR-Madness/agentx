/**
 * ChoiceElement — interactive option buttons. Clicking an option submits it as
 * the user's next turn (`onSubmitChoice`); once answered (or while a turn is in
 * flight) the buttons are disabled. The chosen option is marked `aria-pressed`.
 */

import { Button } from '../../ui/Button';
import type { ElementRenderProps } from './types';

export function ChoiceElement({
  element,
  messageId,
  answeredValue,
  busy,
  onSubmitChoice,
}: ElementRenderProps) {
  if (element.type !== 'choice') return null;
  const answered = answeredValue !== undefined;

  return (
    <div className="flex flex-col gap-2">
      {element.prompt && <div className="text-sm text-fg">{element.prompt}</div>}
      <div className="flex flex-wrap gap-2" role="group" aria-label={element.prompt}>
        {element.options.map((option) => {
          const chosen = answered && option === answeredValue;
          return (
            <Button
              key={option}
              type="button"
              variant={chosen ? 'primary' : 'secondary'}
              size="sm"
              disabled={answered || busy}
              aria-pressed={chosen}
              onClick={() => onSubmitChoice?.(option, messageId)}
            >
              {option}
            </Button>
          );
        })}
      </div>
    </div>
  );
}
