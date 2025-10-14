/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bloomberg: {
          black: '#000000',
          amber: '#FF8800',
          green: '#00FF41',
          red: '#FF0000',
          blue: '#0084FF',
          gray: '#808080',
          dark: '#0A0A0A',
          border: '#333333',
        },
        terminal: {
          bg: '#0A0A0A',
          border: '#333333',
          text: '#00FF41',
          accent: '#FF8800',
          warning: '#FF0000',
          info: '#0084FF',
        }
      },
      fontFamily: {
        'terminal': ['Consolas', 'Monaco', 'Courier New', 'monospace'],
        'bloomberg': ['Consolas', 'Monaco', 'Courier New', 'monospace'],
      },
      fontSize: {
        'terminal-xs': '10px',
        'terminal-sm': '12px',
        'terminal-base': '14px',
        'terminal-lg': '16px',
      },
      animation: {
        'blink': 'blink 1s infinite',
        'data-rain': 'data-rain 2s linear infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        blink: {
          '0%, 50%': { opacity: '1' },
          '51%, 100%': { opacity: '0' },
        },
        'data-rain': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        glow: {
          '0%': { textShadow: '0 0 5px #00FF41' },
          '100%': { textShadow: '0 0 20px #00FF41, 0 0 30px #00FF41' },
        }
      },
      boxShadow: {
        'terminal': '0 0 10px rgba(0, 255, 65, 0.3)',
        'bloomberg': '0 0 20px rgba(255, 136, 0, 0.4)',
      }
    },
  },
  plugins: [],
}
