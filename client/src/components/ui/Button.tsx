/**
 * Button component with variants
 * Built on top of existing button styles with Radix primitives compatibility
 */

import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';

const buttonVariants = cva(
  // Layout/interaction utilities; the brand skin (gradient, shimmer, hover glow)
  // comes from the variant's global class below. Now that Tailwind is enabled
  // and base.css sits in the `base` layer, the size utilities (in `utilities`)
  // correctly out-rank the class's padding, so the `size` prop is effective.
  'inline-flex items-center justify-center gap-2 font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-surface-base',
  {
    variants: {
      variant: {
        primary: 'button-primary',
        secondary: 'button-secondary',
        ghost: 'button-ghost',
        danger: 'button-secondary danger',
      },
      size: {
        // `md` keeps each variant class's native padding (the established look).
        sm: 'px-3 py-1.5 text-sm rounded-lg min-h-[36px]',
        md: '',
        lg: 'px-6 py-3 text-base min-h-[44px]',
        icon: 'p-2 rounded-lg min-h-[36px] min-w-[36px]',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  }
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, loading, disabled, children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        disabled={disabled || loading}
        {...props}
      >
        {loading && (
          <svg
            className="animate-spin h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';
