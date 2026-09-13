import { useEffect } from 'react'
import { toast } from 'sonner'

/** Fires a toast whenever `error` changes to a non-null message, instead of rendering it inline. */
export function useErrorToast(error: string | null | undefined) {
  useEffect(() => {
    if (error) toast.error(error)
  }, [error])
}
