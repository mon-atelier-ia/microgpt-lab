import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogTitle,
} from './alert-dialog';
import { Button } from './button';

export type ConfirmResetDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
  title: string;
  description: string;
};

export function ConfirmResetDialog({
  open,
  onOpenChange,
  onConfirm,
  title,
  description,
}: ConfirmResetDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogTitle className="text-sm font-semibold">{title}</AlertDialogTitle>
        <AlertDialogDescription className="mt-2 text-xs text-text-secondary">
          {description}
        </AlertDialogDescription>
        <div className="mt-4 flex justify-end gap-2">
          <AlertDialogCancel asChild>
            <Button size="sm" variant="ghost">
              Annuler
            </Button>
          </AlertDialogCancel>
          <AlertDialogAction asChild>
            <Button size="sm" onClick={onConfirm} className="bg-error text-surface-0">
              Réinitialiser
            </Button>
          </AlertDialogAction>
        </div>
      </AlertDialogContent>
    </AlertDialog>
  );
}
