import { useEffect } from 'react';

export interface HotkeyConfig {
  key: string;
  ctrl?: boolean;
  alt?: boolean;
  shift?: boolean;
  action: () => void;
  description: string;
}

export const useHotkeys = (hotkeys: HotkeyConfig[]) => {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const matchingHotkey = hotkeys.find(hotkey => {
        const keyMatch = event.key.toLowerCase() === hotkey.key.toLowerCase();
        const ctrlMatch = !!event.ctrlKey === !!hotkey.ctrl;
        const altMatch = !!event.altKey === !!hotkey.alt;
        const shiftMatch = !!event.shiftKey === !!hotkey.shift;
        
        return keyMatch && ctrlMatch && altMatch && shiftMatch;
      });

      if (matchingHotkey) {
        event.preventDefault();
        matchingHotkey.action();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [hotkeys]);
};

// Bloomberg Terminal-style hotkeys
export const bloombergHotkeys: HotkeyConfig[] = [
  {
    key: '1',
    ctrl: true,
    action: () => console.log('Market Overview'),
    description: 'Market Overview'
  },
  {
    key: '2',
    ctrl: true,
    action: () => console.log('Portfolio'),
    description: 'Portfolio'
  },
  {
    key: '3',
    ctrl: true,
    action: () => console.log('Charts'),
    description: 'Charts'
  },
  {
    key: '4',
    ctrl: true,
    action: () => console.log('System Metrics'),
    description: 'System Metrics'
  },
  {
    key: '5',
    ctrl: true,
    action: () => console.log('Activity Log'),
    description: 'Activity Log'
  },
  {
    key: 'r',
    ctrl: true,
    action: () => window.location.reload(),
    description: 'Refresh'
  },
  {
    key: 'f',
    ctrl: true,
    action: () => console.log('Full Screen'),
    description: 'Full Screen'
  },
  {
    key: 'h',
    ctrl: true,
    action: () => console.log('Help'),
    description: 'Help'
  },
  {
    key: 'Escape',
    action: () => console.log('Escape'),
    description: 'Escape'
  },
  {
    key: 'F1',
    action: () => console.log('Help'),
    description: 'Help'
  },
  {
    key: 'F2',
    action: () => console.log('Settings'),
    description: 'Settings'
  },
  {
    key: 'F3',
    action: () => console.log('Search'),
    description: 'Search'
  },
  {
    key: 'F4',
    action: () => console.log('Export'),
    description: 'Export'
  },
  {
    key: 'F5',
    action: () => window.location.reload(),
    description: 'Refresh'
  },
  {
    key: 'F11',
    action: () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    },
    description: 'Full Screen Toggle'
  }
];

export const getHotkeyDisplay = (hotkey: HotkeyConfig): string => {
  const parts: string[] = [];
  
  if (hotkey.ctrl) parts.push('Ctrl');
  if (hotkey.alt) parts.push('Alt');
  if (hotkey.shift) parts.push('Shift');
  
  parts.push(hotkey.key.toUpperCase());
  
  return parts.join(' + ');
};
