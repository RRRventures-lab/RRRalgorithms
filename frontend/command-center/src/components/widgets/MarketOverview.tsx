import React, { useEffect, useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  Grid,
  Chip,
  IconButton,
  Skeleton,
} from '@mui/material';
import { TrendingUp, TrendingDown, Remove, Refresh } from '@mui/icons-material';
import { motion } from 'framer-motion';

interface CryptoPrice {
  symbol: string;
  price: number;
  change24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  sparkline: number[];
}

const mockData: CryptoPrice[] = [
  {
    symbol: 'BTC-USD',
    price: 110768.89,
    change24h: -1.95,
    volume24h: 28500000000,
    high24h: 113000,
    low24h: 108500,
    sparkline: [108500, 109000, 110000, 111500, 110768],
  },
  {
    symbol: 'ETH-USD',
    price: 3750.60,
    change24h: -2.25,
    volume24h: 15600000000,
    high24h: 3850,
    low24h: 3700,
    sparkline: [3700, 3720, 3780, 3800, 3750],
  },
  {
    symbol: 'SOL-USD',
    price: 177.82,
    change24h: -5.83,
    volume24h: 2100000000,
    high24h: 189,
    low24h: 175,
    sparkline: [189, 185, 180, 178, 177],
  },
  {
    symbol: 'ADA-USD',
    price: 0.63,
    change24h: -3.93,
    volume24h: 450000000,
    high24h: 0.66,
    low24h: 0.62,
    sparkline: [0.65, 0.64, 0.635, 0.632, 0.63],
  },
];

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const height = 40;
  const width = 100;
  
  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * width;
    const y = height - ((value - min) / (max - min)) * height;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2"
        points={points}
      />
    </svg>
  );
}

export default function MarketOverview() {
  const [prices, setPrices] = useState<CryptoPrice[]>(mockData);
  const [loading, setLoading] = useState(false);

  const refreshData = async () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setPrices(mockData.map(p => ({
        ...p,
        price: p.price * (1 + (Math.random() - 0.5) * 0.01),
        change24h: p.change24h + (Math.random() - 0.5) * 0.5,
      })));
      setLoading(false);
    }, 1000);
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setPrices(prev => prev.map(p => ({
        ...p,
        price: p.price * (1 + (Math.random() - 0.5) * 0.001),
      })));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUp sx={{ fontSize: 16 }} />;
    if (change < 0) return <TrendingDown sx={{ fontSize: 16 }} />;
    return <Remove sx={{ fontSize: 16 }} />;
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return '#00ff88';
    if (change < 0) return '#ff3366';
    return '#888';
  };

  return (
    <Paper
      sx={{
        p: 2,
        height: '100%',
        background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Market Overview
        </Typography>
        <IconButton size="small" onClick={refreshData} disabled={loading}>
          <Refresh />
        </IconButton>
      </Box>

      <Grid container spacing={2}>
        {prices.map((crypto, index) => (
          <Grid item xs={12} sm={6} md={3} key={crypto.symbol}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  border: '1px solid rgba(255, 255, 255, 0.05)',
                  bgcolor: 'rgba(255, 255, 255, 0.02)',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.05)',
                    borderColor: 'primary.main',
                  },
                  transition: 'all 0.3s ease',
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    {crypto.symbol}
                  </Typography>
                  <Chip
                    size="small"
                    icon={getTrendIcon(crypto.change24h)}
                    label={`${crypto.change24h > 0 ? '+' : ''}${crypto.change24h.toFixed(2)}%`}
                    sx={{
                      bgcolor: getChangeColor(crypto.change24h) + '20',
                      color: getChangeColor(crypto.change24h),
                      borderColor: getChangeColor(crypto.change24h),
                      '& .MuiChip-icon': {
                        color: getChangeColor(crypto.change24h),
                      },
                    }}
                    variant="outlined"
                  />
                </Box>

                {loading ? (
                  <Skeleton variant="text" width="100%" height={32} />
                ) : (
                  <Typography
                    variant="h5"
                    sx={{
                      fontWeight: 700,
                      color: getChangeColor(crypto.change24h),
                      fontFamily: 'monospace',
                    }}
                  >
                    ${crypto.price.toLocaleString('en-US', {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: crypto.price < 1 ? 4 : 2,
                    })}
                  </Typography>
                )}

                <Box sx={{ mt: 1, mb: 1 }}>
                  <Sparkline
                    data={crypto.sparkline}
                    color={getChangeColor(crypto.change24h)}
                  />
                </Box>

                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      24h High
                    </Typography>
                    <Typography variant="body2">
                      ${crypto.high24h.toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      24h Low
                    </Typography>
                    <Typography variant="body2">
                      ${crypto.low24h.toLocaleString()}
                    </Typography>
                  </Grid>
                </Grid>

                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    24h Volume
                  </Typography>
                  <Typography variant="body2">
                    ${(crypto.volume24h / 1e9).toFixed(2)}B
                  </Typography>
                </Box>
              </Box>
            </motion.div>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
}