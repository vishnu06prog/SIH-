import type { ReactNode } from "react";
import { InboxIcon } from "./Icons";

interface Props {
  title: string;
  description?: string | null;
  icon?: ReactNode;
  action?: ReactNode;
}

export function EmptyState({ title, description, icon, action }: Props) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon ?? <InboxIcon size={24} />}</div>
      <h3>{title}</h3>
      {description ? <p>{description}</p> : null}
      {action}
    </div>
  );
}
