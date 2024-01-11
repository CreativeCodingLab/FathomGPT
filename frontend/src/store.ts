import { writable } from 'svelte/store';
import type { speciesData } from '$lib/types/responseType';


type ActiveImageStoreType = {
    isImageDetailsOpen: boolean;
    species: speciesData | null;
  };

export const activeImageStore = writable<ActiveImageStoreType>({
  isImageDetailsOpen: false,
  species: null
});