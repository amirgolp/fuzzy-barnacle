import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface FavoritesState {
  favorites: string[]
  addFavorite: (symbol: string) => void
  removeFavorite: (symbol: string) => void
  toggleFavorite: (symbol: string) => void
  isFavorite: (symbol: string) => boolean
}

export const useFavoritesStore = create<FavoritesState>()(
  persist(
    (set, get) => ({
      favorites: [],
      addFavorite: (symbol) =>
        set((state) => ({
          favorites: state.favorites.includes(symbol)
            ? state.favorites
            : [...state.favorites, symbol],
        })),
      removeFavorite: (symbol) =>
        set((state) => ({
          favorites: state.favorites.filter((s) => s !== symbol),
        })),
      toggleFavorite: (symbol) =>
        set((state) => {
          const isFav = state.favorites.includes(symbol)
          return {
            favorites: isFav
              ? state.favorites.filter((s) => s !== symbol)
              : [...state.favorites, symbol],
          }
        }),
      isFavorite: (symbol) => get().favorites.includes(symbol),
    }),
    {
      name: 'favorites-storage', // unique name for localStorage key
    },
  ),
)
