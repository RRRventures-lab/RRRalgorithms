from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import SwiftUI
import aiohttp
import asyncio
import json
import logging
import os

#!/usr/bin/env python
"""
Figma MCP Integration for Design Collaboration
==============================================

This module provides integration with Figma for design collaboration,
allowing the trading app design to be synchronized between Figma and code.
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FigmaColor:
    """Represents a color in the design system"""
    name: str
    value: str
    opacity: float = 1.0
    description: str = ""
    
    def to_swift(self) -> str:
        """Convert to SwiftUI Color"""
        return f'Color(hex: "{self.value}")'
    
    def to_css(self) -> str:
        """Convert to CSS color"""
        if self.opacity < 1.0:
            return f'rgba({self.value}, {self.opacity})'
        return self.value


@dataclass
class FigmaTypography:
    """Represents typography style"""
    name: str
    font_family: str
    font_size: int
    font_weight: str
    line_height: float
    letter_spacing: float = 0
    
    def to_swift(self) -> str:
        """Convert to SwiftUI Font"""
        weight_map = {
            'regular': '.regular',
            'medium': '.medium',
            'semibold': '.semibold',
            'bold': '.bold',
            'heavy': '.heavy'
        }
        weight = weight_map.get(self.font_weight.lower(), '.regular')
        return f'.custom("{self.font_family}", size: {self.font_size}, weight: {weight})'
    
    def to_css(self) -> Dict[str, str]:
        """Convert to CSS styles"""
        return {
            'font-family': self.font_family,
            'font-size': f'{self.font_size}px',
            'font-weight': self.font_weight,
            'line-height': str(self.line_height),
            'letter-spacing': f'{self.letter_spacing}px' if self.letter_spacing else '0'
        }


@dataclass
class FigmaComponent:
    """Represents a reusable component"""
    name: str
    type: str
    properties: Dict[str, Any]
    children: List['FigmaComponent'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class FigmaDesignSystem:
    """Manages the design system from Figma"""
    
    def __init__(self):
        self.colors: Dict[str, FigmaColor] = {}
        self.typography: Dict[str, FigmaTypography] = {}
        self.components: Dict[str, FigmaComponent] = {}
        self.spacing = {}
        self.breakpoints = {}
        
        # Initialize with trading app design system
        self._initialize_design_system()
    
    def _initialize_design_system(self):
        """Initialize with the trading app design system"""
        # Colors
        self.colors = {
            'primary-green': FigmaColor('Primary Green', '#00FF88', description='Profits, buys, success'),
            'error-red': FigmaColor('Error Red', '#FF3366', description='Losses, sells, errors'),
            'warning-orange': FigmaColor('Warning Orange', '#FFAA00', description='Warnings, paused states'),
            'info-blue': FigmaColor('Info Blue', '#00AAFF', description='Information, neutral'),
            'background': FigmaColor('Background', '#0A0A0A', description='Dark theme base'),
            'surface': FigmaColor('Surface', '#1A1A1A', description='Card backgrounds'),
            'surface-light': FigmaColor('Surface Light', '#2A2A2A', description='Elevated surfaces'),
            'text-primary': FigmaColor('Text Primary', '#FFFFFF', description='Primary text'),
            'text-secondary': FigmaColor('Text Secondary', '#999999', description='Secondary text'),
        }
        
        # Typography
        self.typography = {
            'heading-1': FigmaTypography('Heading 1', 'SF Pro Display', 36, 'bold', 1.2),
            'heading-2': FigmaTypography('Heading 2', 'SF Pro Display', 28, 'bold', 1.3),
            'heading-3': FigmaTypography('Heading 3', 'SF Pro Display', 24, 'semibold', 1.3),
            'body-large': FigmaTypography('Body Large', 'SF Pro Text', 18, 'regular', 1.5),
            'body': FigmaTypography('Body', 'SF Pro Text', 16, 'regular', 1.5),
            'body-small': FigmaTypography('Body Small', 'SF Pro Text', 14, 'regular', 1.4),
            'caption': FigmaTypography('Caption', 'SF Pro Text', 12, 'regular', 1.3),
            'mono': FigmaTypography('Monospace', 'SF Mono', 14, 'regular', 1.4),
        }
        
        # Spacing
        self.spacing = {
            'xs': 4,
            'sm': 8,
            'md': 16,
            'lg': 24,
            'xl': 32,
            'xxl': 48
        }
        
        # Breakpoints
        self.breakpoints = {
            'mobile': 375,
            'tablet': 768,
            'desktop': 1024,
            'wide': 1440
        }
    
    def export_to_swift(self) -> str:
        """Export design system to SwiftUI"""
        swift_code = """
// Generated Design System from Figma
// Last updated: {timestamp}


// MARK: - Colors
extension Color {
"""
        # Add colors
        for key, color in self.colors.items():
            var_name = key.replace('-', '_')
            swift_code += f'    static let {var_name} = {color.to_swift()}\n'
        
        swift_code += """
}

// MARK: - Typography
extension Font {
"""
        # Add typography
        for key, typo in self.typography.items():
            var_name = key.replace('-', '_')
            swift_code += f'    static let {var_name} = Font{typo.to_swift()}\n'
        
        swift_code += """
}

// MARK: - Spacing
struct Spacing {
"""
        # Add spacing
        for key, value in self.spacing.items():
            swift_code += f'    static let {key}: CGFloat = {value}\n'
        
        swift_code += "}\n"
        
        return swift_code.format(timestamp=datetime.now().isoformat())
    
    def export_to_css(self) -> str:
        """Export design system to CSS"""
        css_code = """
/* Generated Design System from Figma */
/* Last updated: {timestamp} */

:root {
"""
        # Add colors
        for key, color in self.colors.items():
            css_code += f'  --color-{key}: {color.to_css()};\n'
        
        # Add spacing
        for key, value in self.spacing.items():
            css_code += f'  --spacing-{key}: {value}px;\n'
        
        # Add breakpoints
        for key, value in self.breakpoints.items():
            css_code += f'  --breakpoint-{key}: {value}px;\n'
        
        css_code += "}\n\n"
        
        # Add typography classes
        for key, typo in self.typography.items():
            css_code += f".text-{key} {{\n"
            for prop, value in typo.to_css().items():
                css_code += f"  {prop}: {value};\n"
            css_code += "}\n\n"
        
        return css_code.format(timestamp=datetime.now().isoformat())


class FigmaMCPServer:
    """MCP Server for Figma integration"""
    
    def __init__(self, api_key: str = None, file_key: str = None):
        self.api_key = api_key or os.getenv('FIGMA_API_KEY')
        self.file_key = file_key or os.getenv('FIGMA_FILE_KEY')
        self.base_url = "https://api.figma.com/v1"
        self.design_system = FigmaDesignSystem()
        self.session: Optional[aiohttp.ClientSession] = None
        
        if not self.api_key:
            logger.warning("Figma API key not provided. Using local design system only.")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={'X-Figma-Token': self.api_key} if self.api_key else {}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_file(self) -> Dict[str, Any]:
        """Fetch the Figma file"""
        if not self.api_key or not self.file_key:
            return {}
        
        url = f"{self.base_url}/files/{self.file_key}"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Failed to fetch Figma file: {response.status}")
                return {}
    
    async def fetch_styles(self) -> Dict[str, Any]:
        """Fetch styles from Figma"""
        if not self.api_key or not self.file_key:
            return {}
        
        url = f"{self.base_url}/files/{self.file_key}/styles"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Failed to fetch Figma styles: {response.status}")
                return {}
    
    async def fetch_components(self) -> Dict[str, Any]:
        """Fetch components from Figma"""
        if not self.api_key or not self.file_key:
            return {}
        
        url = f"{self.base_url}/files/{self.file_key}/components"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Failed to fetch Figma components: {response.status}")
                return {}
    
    async def sync_design_system(self):
        """Sync design system from Figma"""
        if not self.api_key:
            logger.info("Using local design system (no Figma API key)")
            return
        
        logger.info("Syncing design system from Figma...")
        
        # Fetch data from Figma
        # Optimized: Parallel execution
file_data, styles = await asyncio.gather(
    self.fetch_file(),
    self.fetch_styles()
)
        components = await self.fetch_components()
        
        # Process and update design system
        if file_data:
            self._process_file_data(file_data)
        
        if styles:
            self._process_styles(styles)
        
        if components:
            self._process_components(components)
        
        logger.info("Design system sync complete")
    
    def _process_file_data(self, data: Dict[str, Any]):
        """Process file data from Figma"""
        # Extract design tokens from file
        pass
    
    def _process_styles(self, data: Dict[str, Any]):
        """Process styles from Figma"""
        # Extract colors and typography
        pass
    
    def _process_components(self, data: Dict[str, Any]):
        """Process components from Figma"""
        # Extract reusable components
        pass
    
    async def export_design_system(self, format: str = 'all'):
        """Export the design system"""
        outputs = {}
        
        if format in ['all', 'swift']:
            swift_code = self.design_system.export_to_swift()
            swift_path = Path('ios/TradingCommand/TradingCommand/DesignSystem.swift')
            swift_path.parent.mkdir(parents=True, exist_ok=True)
            swift_path.write_text(swift_code)
            outputs['swift'] = str(swift_path)
            logger.info(f"Exported Swift design system to {swift_path}")
        
        if format in ['all', 'css']:
            css_code = self.design_system.export_to_css()
            css_path = Path('frontend/command-center/src/styles/design-system.css')
            css_path.parent.mkdir(parents=True, exist_ok=True)
            css_path.write_text(css_code)
            outputs['css'] = str(css_path)
            logger.info(f"Exported CSS design system to {css_path}")
        
        if format in ['all', 'json']:
            json_data = {
                'colors': {k: asdict(v) for k, v in self.design_system.colors.items()},
                'typography': {k: asdict(v) for k, v in self.design_system.typography.items()},
                'spacing': self.design_system.spacing,
                'breakpoints': self.design_system.breakpoints
            }
            json_path = Path('src/design/design-tokens.json')
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(json_data, indent=2))
            outputs['json'] = str(json_path)
            logger.info(f"Exported JSON design tokens to {json_path}")
        
        return outputs
    
    async def create_component(self, name: str, type: str, properties: Dict[str, Any]):
        """Create a new component in Figma"""
        component = FigmaComponent(name, type, properties)
        self.design_system.components[name] = component
        logger.info(f"Created component: {name}")
        return component
    
    async def update_component(self, name: str, properties: Dict[str, Any]):
        """Update an existing component"""
        if name in self.design_system.components:
            component = self.design_system.components[name]
            component.properties.update(properties)
            logger.info(f"Updated component: {name}")
            return component
        else:
            logger.error(f"Component not found: {name}")
            return None


class FigmaDesignPreview:
    """Generate preview of designs for mobile app"""
    
    def __init__(self, design_system: FigmaDesignSystem):
        self.design_system = design_system
    
    def generate_preview_html(self) -> str:
        """Generate HTML preview of the design"""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading App Design Preview</title>
    <style>
        {css_styles}
        
        body {
            margin: 0;
            padding: 20px;
            background: var(--color-background);
            color: var(--color-text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        }
        
        .preview-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .section {
            margin-bottom: 40px;
            padding: 20px;
            background: var(--color-surface);
            border-radius: 12px;
        }
        
        .color-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }
        
        .color-swatch {
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        
        .typography-sample {
            margin: 16px 0;
            padding: 12px;
            background: var(--color-surface-light);
            border-radius: 8px;
        }
        
        .component-preview {
            margin: 16px 0;
            padding: 20px;
            background: var(--color-surface-light);
            border-radius: 8px;
        }
        
        .mobile-preview {
            width: 375px;
            height: 812px;
            margin: 20px auto;
            border: 16px solid #333;
            border-radius: 36px;
            overflow: hidden;
            background: var(--color-background);
        }
    </style>
</head>
<body>
    <div class="preview-container">
        <h1>Trading App Design System</h1>
        
        <!-- Colors Section -->
        <div class="section">
            <h2>Colors</h2>
            <div class="color-grid">
                {color_swatches}
            </div>
        </div>
        
        <!-- Typography Section -->
        <div class="section">
            <h2>Typography</h2>
            {typography_samples}
        </div>
        
        <!-- Components Section -->
        <div class="section">
            <h2>Components</h2>
            {component_previews}
        </div>
        
        <!-- Mobile Preview -->
        <div class="section">
            <h2>Mobile Preview</h2>
            <div class="mobile-preview">
                {mobile_content}
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        # Generate CSS styles
        css_styles = self.design_system.export_to_css()
        
        # Generate color swatches
        color_swatches = ""
        for key, color in self.design_system.colors.items():
            color_swatches += f"""
            <div class="color-swatch" style="background: {color.value};">
                <div style="color: {'#000' if key.startswith('primary') else '#fff'};">
                    <strong>{color.name}</strong><br>
                    <small>{color.value}</small><br>
                    <small>{color.description}</small>
                </div>
            </div>
            """
        
        # Generate typography samples
        typography_samples = ""
        for key, typo in self.design_system.typography.items():
            typography_samples += f"""
            <div class="typography-sample text-{key}">
                <strong>{typo.name}</strong> - {typo.font_family} {typo.font_size}px {typo.font_weight}<br>
                <span>The quick brown fox jumps over the lazy dog</span>
            </div>
            """
        
        # Generate component previews
        component_previews = self._generate_component_previews()
        
        # Generate mobile content
        mobile_content = self._generate_mobile_preview()
        
        return html.format(
            css_styles=css_styles,
            color_swatches=color_swatches,
            typography_samples=typography_samples,
            component_previews=component_previews,
            mobile_content=mobile_content
        )
    
    def _generate_component_previews(self) -> str:
        """Generate component preview HTML"""
        return """
        <div class="component-preview">
            <h3>Trading Card</h3>
            <div style="padding: 16px; background: var(--color-surface); border-radius: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <div>
                        <div style="font-weight: 600;">BTC-USD</div>
                        <div style="font-size: 12px; color: var(--color-text-secondary);">Bitcoin</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 600;">$110,768.89</div>
                        <div style="color: var(--color-primary-green); font-size: 12px;">+2.34%</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="component-preview">
            <h3>Action Buttons</h3>
            <div style="display: flex; gap: 12px;">
                <button style="flex: 1; padding: 12px; background: var(--color-primary-green); color: white; border: none; border-radius: 8px; font-weight: 600;">Buy</button>
                <button style="flex: 1; padding: 12px; background: var(--color-error-red); color: white; border: none; border-radius: 8px; font-weight: 600;">Sell</button>
            </div>
        </div>
        """
    
    def _generate_mobile_preview(self) -> str:
        """Generate mobile preview content"""
        return """
        <div style="padding: 20px;">
            <!-- Header -->
            <div style="margin-bottom: 20px;">
                <h2 style="margin: 0;">Portfolio</h2>
                <div style="font-size: 32px; font-weight: bold; color: var(--color-primary-green);">$125,430.50</div>
                <div style="color: var(--color-primary-green);">+$2,340.50 (+1.90%) Today</div>
            </div>
            
            <!-- Positions -->
            <div style="background: var(--color-surface); padding: 16px; border-radius: 12px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <div>BTC-USD</div>
                    <div style="color: var(--color-primary-green);">+$1,384.45</div>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <div style="color: var(--color-text-secondary); font-size: 14px;">0.5 BTC</div>
                    <div style="color: var(--color-text-secondary); font-size: 14px;">+2.50%</div>
                </div>
            </div>
            
            <!-- Action Buttons -->
            <div style="position: fixed; bottom: 40px; left: 20px; right: 20px; display: flex; gap: 12px;">
                <button style="flex: 1; padding: 16px; background: var(--color-primary-green); color: white; border: none; border-radius: 12px; font-weight: 600; font-size: 16px;">Buy</button>
                <button style="flex: 1; padding: 16px; background: var(--color-error-red); color: white; border: none; border-radius: 12px; font-weight: 600; font-size: 16px;">Sell</button>
            </div>
        </div>
        """
    
    def save_preview(self, output_path: str = 'design_preview.html'):
        """Save the preview to a file"""
        html_content = self.generate_preview_html()
        path = Path(output_path)
        path.write_text(html_content)
        logger.info(f"Design preview saved to {path}")
        return str(path)


async def main():
    """Main function to demonstrate Figma integration"""
    
    # Create MCP server
    async with FigmaMCPServer() as server:
        # Sync design system from Figma
        await server.sync_design_system()
        
        # Export design system
        outputs = await server.export_design_system()
        print(f"Exported design system: {outputs}")
        
        # Create design preview
        preview = FigmaDesignPreview(server.design_system)
        preview_path = preview.save_preview('docs/design/preview.html')
        print(f"Design preview saved to: {preview_path}")
        
        # Create sample components
        await server.create_component(
            "TradingCard",
            "card",
            {
                "symbol": "BTC-USD",
                "price": 110768.89,
                "change": 2.34,
                "volume": 28.5
            }
        )
        
        await server.create_component(
            "PortfolioSummary",
            "summary",
            {
                "totalValue": 125430.50,
                "dailyPnL": 2340.50,
                "positions": 5
            }
        )


if __name__ == "__main__":
    asyncio.run(main()