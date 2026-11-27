import { cn } from '@/lib/utils'

// A row of mutually exclusive choices, styled like the header's tabs.
const Segmented = <T extends string>({
  value,
  options,
  label,
  onChange
}: {
  value: T
  options: { value: T; label: string }[]
  label: string
  onChange: (value: T) => void
}) => (
  <div
    role="tablist"
    aria-label={label}
    className="bg-muted text-muted-foreground inline-flex h-8 items-center rounded-md p-1"
  >
    {options.map((option) => (
      <button
        key={option.value}
        type="button"
        role="tab"
        aria-selected={value === option.value}
        className={cn(
          'inline-flex cursor-pointer items-center rounded-sm px-2 py-1 text-sm font-medium whitespace-nowrap transition-all',
          value === option.value ? 'bg-background text-foreground shadow-sm' : 'hover:bg-background/60'
        )}
        onClick={() => onChange(option.value)}
      >
        {option.label}
      </button>
    ))}
  </div>
)

export default Segmented
