# 🎨 Figma MCP Integration Setup Guide

Complete guide for setting up Figma design collaboration with the trading command center.

## 📋 Prerequisites

- Figma account (Professional or higher for API access)
- Figma API token
- Trading app Figma file

## 🔑 Getting Your Figma API Token

1. **Log into Figma**
   - Go to [figma.com](https://figma.com)
   - Sign in to your account

2. **Access Account Settings**
   - Click your profile picture (top-left)
   - Select "Settings"

3. **Generate Personal Access Token**
   - Scroll to "Personal access tokens"
   - Click "Create new token"
   - Name it: "Trading App MCP"
   - Copy the token immediately (shown only once!)

## 📁 Setting Up Your Figma File

### 1. Create Trading App File

```
Trading Command Center
├── 📱 Mobile Screens
│   ├── Dashboard
│   ├── Trading
│   ├── Portfolio
│   ├── Backtesting
│   └── Settings
├── 🎨 Design System
│   ├── Colors
│   ├── Typography
│   ├── Components
│   └── Icons
└── 🔄 Prototypes
    ├── User Flow
    └── Interactions
```

### 2. Get File Key

1. Open your Figma file
2. Look at the URL: `https://www.figma.com/file/[FILE_KEY]/[FILE_NAME]`
3. Copy the FILE_KEY portion

## 🔧 Configure MCP Server

### 1. Environment Variables

Create `.env` file in project root:

```bash
# Figma Configuration
FIGMA_API_KEY=your-figma-api-token-here
FIGMA_FILE_KEY=your-file-key-here
FIGMA_TEAM_ID=optional-team-id
```

### 2. Install Dependencies

```bash
pip install aiohttp
pip install python-dotenv
```

### 3. Test Connection

```bash
python src/design/figma_integration.py
```

## 🎨 Design System Structure

### Colors
```
Primary
├── Green (#00FF88) - Profits, Success
├── Red (#FF3366) - Losses, Errors
├── Blue (#00AAFF) - Information
└── Orange (#FFAA00) - Warnings

Neutral
├── Background (#0A0A0A)
├── Surface (#1A1A1A)
├── Surface Light (#2A2A2A)
├── Text Primary (#FFFFFF)
└── Text Secondary (#999999)
```

### Typography
```
Headings
├── H1 - SF Pro Display Bold 36px
├── H2 - SF Pro Display Bold 28px
└── H3 - SF Pro Display Semibold 24px

Body
├── Large - SF Pro Text 18px
├── Regular - SF Pro Text 16px
├── Small - SF Pro Text 14px
└── Caption - SF Pro Text 12px

Special
└── Monospace - SF Mono 14px
```

### Components Library

#### Trading Cards
```figma
Component: TradingCard
├── Symbol (text)
├── Price (number)
├── Change (percentage)
├── Volume (number)
└── Chart (sparkline)
```

#### Control Buttons
```figma
Component: ActionButton
├── Primary (Buy - Green)
├── Danger (Sell - Red)
├── Warning (Pause - Orange)
└── Info (Details - Blue)
```

#### Charts
```figma
Component: TradingChart
├── Candlestick
├── Line Chart
├── Area Chart
└── Bar Chart
```

## 🔄 Syncing Design with Code

### Auto-Sync Workflow

1. **Design in Figma** → Make changes to components
2. **Run Sync Script** → `python src/design/figma_integration.py`
3. **Generated Files**:
   - `ios/TradingCommand/DesignSystem.swift` - iOS styles
   - `frontend/src/styles/design-system.css` - Web styles
   - `src/design/design-tokens.json` - Design tokens

### Manual Sync

```python
from src.design.figma_integration import FigmaMCPServer

async def sync_design():
    async with FigmaMCPServer() as server:
        await server.sync_design_system()
        await server.export_design_system('all')
```

## 📱 iOS Integration

### Using Design System in SwiftUI

```swift
import SwiftUI

struct TradingCard: View {
    var body: some View {
        VStack {
            Text("BTC-USD")
                .font(.heading_3)
                .foregroundColor(.text_primary)
            
            Text("$110,768.89")
                .font(.mono)
                .foregroundColor(.primary_green)
        }
        .padding(Spacing.md)
        .background(Color.surface)
        .cornerRadius(12)
    }
}
```

## 💻 Web Integration

### Using Design System in React

```jsx
import './styles/design-system.css';

function TradingCard({ symbol, price, change }) {
  return (
    <div className="card" style={{ background: 'var(--color-surface)' }}>
      <h3 className="text-heading-3">{symbol}</h3>
      <div className="text-mono" style={{ color: 'var(--color-primary-green)' }}>
        ${price.toLocaleString()}
      </div>
    </div>
  );
}
```

## 🖼️ Preview System

### Generate Design Preview

```bash
python -c "
from src.design.figma_integration import FigmaDesignPreview, FigmaDesignSystem
preview = FigmaDesignPreview(FigmaDesignSystem())
preview.save_preview('docs/design/preview.html')
"
```

### View Preview

```bash
open docs/design/preview.html
```

## 🤝 Collaboration Workflow

### For Designers

1. **Create/Update in Figma**
   - Work on components in Figma
   - Use consistent naming conventions
   - Follow design system guidelines

2. **Publish Changes**
   - Publish library updates
   - Add version notes
   - Notify developers

### For Developers

1. **Sync Latest Design**
   ```bash
   npm run sync:design
   # or
   python src/design/figma_integration.py
   ```

2. **Review Changes**
   - Check generated files
   - Test in app
   - Report issues

3. **Implement Components**
   - Use design tokens
   - Follow component structure
   - Maintain consistency

## 📊 Design Metrics

### Track Design System Usage

```python
# Analytics for design system adoption
{
    "colors_used": ["primary-green", "error-red"],
    "typography_used": ["heading-1", "body"],
    "components_created": 42,
    "sync_frequency": "daily",
    "consistency_score": 0.92
}
```

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check API token validity
   - Verify network connection
   - Ensure Figma file permissions

2. **Sync Not Working**
   - Validate file key
   - Check component naming
   - Review console logs

3. **Styles Not Applying**
   - Clear cache
   - Rebuild app
   - Verify import statements

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python src/design/figma_integration.py --verbose
```

## 🚀 Advanced Features

### Component Variants

```figma
Button Component
├── State
│   ├── Default
│   ├── Hover
│   ├── Pressed
│   └── Disabled
├── Size
│   ├── Small
│   ├── Medium
│   └── Large
└── Type
    ├── Primary
    ├── Secondary
    └── Tertiary
```

### Responsive Design

```css
/* Breakpoints from Figma */
@media (min-width: var(--breakpoint-mobile)) { }
@media (min-width: var(--breakpoint-tablet)) { }
@media (min-width: var(--breakpoint-desktop)) { }
```

### Dark/Light Mode

```swift
// Automatic theme switching
struct ThemedView: View {
    @Environment(\.colorScheme) var colorScheme
    
    var backgroundColor: Color {
        colorScheme == .dark ? .background : .white
    }
}
```

## 📝 Best Practices

1. **Naming Conventions**
   - Use kebab-case for tokens
   - Descriptive component names
   - Consistent property names

2. **Version Control**
   - Commit generated files
   - Tag design system versions
   - Document breaking changes

3. **Documentation**
   - Component usage examples
   - Design rationale
   - Migration guides

## 🔗 Resources

- [Figma API Documentation](https://www.figma.com/developers/api)
- [Design Tokens Spec](https://design-tokens.github.io/community-group/format/)
- [MCP Protocol Spec](https://modelcontextprotocol.io)
- [SwiftUI Documentation](https://developer.apple.com/documentation/swiftui)

---

**Last Updated**: 2025-10-12
**Design System Version**: 1.0.0