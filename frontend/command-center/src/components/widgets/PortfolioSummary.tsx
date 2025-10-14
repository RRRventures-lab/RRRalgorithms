import React from 'react';
import {
  Paper,
  Box,
  Typography,
  Grid,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  AccountBalanceWallet,
  TrendingUp,
  TrendingDown,
  AccessTime,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

interface PortfolioData {
  totalValue: number;
  cash: number;
  invested: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  totalPnL: number;
  totalPnLPercent: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
}

const mockPortfolio: PortfolioData = {
  totalValue: 125430.50,
  cash: 45230.25,
  invested: 80200.25,
  dailyPnL: 2340.50,
  dailyPnLPercent: 1.90,
  totalPnL: 25430.50,
  totalPnLPercent: 25.43,
  winRate: 68.5,
  sharpeRatio: 1.85,
  maxDrawdown: -8.5,
};

export default function PortfolioSummary() {
  const getPnLColor = (value: number) => {
    return value >= 0 ? '#00ff88' : '#ff3366';
  };

  const StatCard = ({ 
    label, 
    value, 
    subValue, 
    icon, 
    color = '#fff' 
  }: {
    label: string;
    value: string;
    subValue?: string;
    icon?: React.ReactNode;
    color?: string;
  }) => (
    <Box>
      <Typography variant="caption" color="text.secondary" gutterBottom>
        {label}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        {icon}
        <Typography
          variant="h6"
          sx={{
            fontWeight: 600,
            color,
            fontFamily: 'monospace',
          }}
        >
          {value}
        </Typography>
      </Box>
      {subValue && (
        <Typography variant="caption" color="text.secondary">
          {subValue}
        </Typography>
      )}
    </Box>
  );

  return (
    <Paper
      sx={{
        p: 2,
        height: '100%',
        background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Animated background gradient */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `linear-gradient(135deg, ${getPnLColor(mockPortfolio.dailyPnL)}10 0%, transparent 100%)`,
          pointerEvents: 'none',
        }}
      />

      <Box sx={{ position: 'relative', zIndex: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AccountBalanceWallet color="primary" />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Portfolio Summary
            </Typography>
          </Box>
          <Chip
            size="small"
            icon={<AccessTime />}
            label="Live"
            color="success"
            sx={{ animation: 'pulse 2s infinite' }}
          />
        </Box>

        <Grid container spacing={2}>
          {/* Total Value */}
          <Grid item xs={12}>
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  bgcolor: 'rgba(0, 255, 136, 0.05)',
                  border: '1px solid rgba(0, 255, 136, 0.2)',
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  Total Portfolio Value
                </Typography>
                <Typography
                  variant="h4"
                  sx={{
                    fontWeight: 700,
                    color: '#00ff88',
                    fontFamily: 'monospace',
                  }}
                >
                  ${mockPortfolio.totalValue.toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                  {mockPortfolio.dailyPnL >= 0 ? (
                    <TrendingUp sx={{ color: '#00ff88', fontSize: 20 }} />
                  ) : (
                    <TrendingDown sx={{ color: '#ff3366', fontSize: 20 }} />
                  )}
                  <Typography
                    variant="body2"
                    sx={{ color: getPnLColor(mockPortfolio.dailyPnL) }}
                  >
                    ${Math.abs(mockPortfolio.dailyPnL).toFixed(2)} (
                    {mockPortfolio.dailyPnLPercent > 0 ? '+' : ''}
                    {mockPortfolio.dailyPnLPercent.toFixed(2)}%) Today
                  </Typography>
                </Box>
              </Box>
            </motion.div>
          </Grid>

          {/* Cash vs Invested */}
          <Grid item xs={6}>
            <StatCard
              label="Cash Available"
              value={`$${(mockPortfolio.cash / 1000).toFixed(1)}K`}
              subValue={`${((mockPortfolio.cash / mockPortfolio.totalValue) * 100).toFixed(0)}% of portfolio`}
              color="#00aaff"
            />
          </Grid>
          <Grid item xs={6}>
            <StatCard
              label="Invested"
              value={`$${(mockPortfolio.invested / 1000).toFixed(1)}K`}
              subValue={`${((mockPortfolio.invested / mockPortfolio.totalValue) * 100).toFixed(0)}% of portfolio`}
              color="#ffaa00"
            />
          </Grid>

          {/* Total P&L */}
          <Grid item xs={12}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Total P&L
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 2 }}>
                <Typography
                  variant="h5"
                  sx={{
                    fontWeight: 600,
                    color: getPnLColor(mockPortfolio.totalPnL),
                    fontFamily: 'monospace',
                  }}
                >
                  ${mockPortfolio.totalPnL.toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                  })}
                </Typography>
                <Chip
                  size="small"
                  label={`${mockPortfolio.totalPnLPercent > 0 ? '+' : ''}${mockPortfolio.totalPnLPercent.toFixed(2)}%`}
                  sx={{
                    bgcolor: getPnLColor(mockPortfolio.totalPnL) + '20',
                    color: getPnLColor(mockPortfolio.totalPnL),
                    borderColor: getPnLColor(mockPortfolio.totalPnL),
                  }}
                  variant="outlined"
                />
              </Box>
            </Box>
          </Grid>

          {/* Performance Metrics */}
          <Grid item xs={4}>
            <StatCard
              label="Win Rate"
              value={`${mockPortfolio.winRate.toFixed(1)}%`}
              color={mockPortfolio.winRate > 50 ? '#00ff88' : '#ff3366'}
            />
          </Grid>
          <Grid item xs={4}>
            <StatCard
              label="Sharpe"
              value={mockPortfolio.sharpeRatio.toFixed(2)}
              color={mockPortfolio.sharpeRatio > 1 ? '#00ff88' : '#ffaa00'}
            />
          </Grid>
          <Grid item xs={4}>
            <StatCard
              label="Max DD"
              value={`${mockPortfolio.maxDrawdown.toFixed(1)}%`}
              color="#ff3366"
            />
          </Grid>

          {/* Portfolio Allocation Bar */}
          <Grid item xs={12}>
            <Box>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Portfolio Allocation
              </Typography>
              <Box sx={{ mt: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={(mockPortfolio.invested / mockPortfolio.totalValue) * 100}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    bgcolor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      background: 'linear-gradient(90deg, #00ff88 0%, #00aaff 100%)',
                      borderRadius: 4,
                    },
                  }}
                />
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    Cash: {((mockPortfolio.cash / mockPortfolio.totalValue) * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Invested: {((mockPortfolio.invested / mockPortfolio.totalValue) * 100).toFixed(0)}%
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
}