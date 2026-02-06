"""Symbol universe configuration with metadata for all supported asset types."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AssetType(Enum):
    """Supported asset types."""
    STOCK = "stock"
    ETF = "etf"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"
    BOND = "bond"


@dataclass(frozen=True)
class SymbolInfo:
    """Symbol metadata."""
    symbol: str
    name: str
    asset_type: AssetType
    exchange: str
    currency: str = "USD"
    description: Optional[str] = None


# =============================================================================
# STOCKS - Major US equities
# =============================================================================
STOCKS: list[SymbolInfo] = [
    SymbolInfo("AAPL", "Apple Inc.", AssetType.STOCK, "NASDAQ", description="Technology - Consumer Electronics"),
    SymbolInfo("MSFT", "Microsoft Corporation", AssetType.STOCK, "NASDAQ", description="Technology - Software"),
    SymbolInfo("GOOGL", "Alphabet Inc.", AssetType.STOCK, "NASDAQ", description="Technology - Internet Services"),
    SymbolInfo("AMZN", "Amazon.com Inc.", AssetType.STOCK, "NASDAQ", description="Consumer - E-commerce"),
    SymbolInfo("TSLA", "Tesla Inc.", AssetType.STOCK, "NASDAQ", description="Automotive - Electric Vehicles"),
    SymbolInfo("NVDA", "NVIDIA Corporation", AssetType.STOCK, "NASDAQ", description="Technology - Semiconductors"),
    SymbolInfo("BRK.B", "Berkshire Hathaway Inc.", AssetType.STOCK, "NYSE", description="Financials - Conglomerate"),
    SymbolInfo("JPM", "JPMorgan Chase & Co.", AssetType.STOCK, "NYSE", description="Financials - Banking"),
    SymbolInfo("V", "Visa Inc.", AssetType.STOCK, "NYSE", description="Financials - Payments"),
    SymbolInfo("WMT", "Walmart Inc.", AssetType.STOCK, "NYSE", description="Consumer - Retail"),
]

# =============================================================================
# ETFs/FUNDS - Exchange Traded Funds
# =============================================================================
FUNDS: list[SymbolInfo] = [
    SymbolInfo("SPY", "SPDR S&P 500 ETF Trust", AssetType.ETF, "NYSE ARCA", description="Tracks S&P 500 Index"),
    SymbolInfo("QQQ", "Invesco QQQ Trust", AssetType.ETF, "NASDAQ", description="Tracks NASDAQ-100 Index"),
    SymbolInfo("VTI", "Vanguard Total Stock Market ETF", AssetType.ETF, "NYSE ARCA", description="Total US Stock Market"),
    SymbolInfo("VOO", "Vanguard S&P 500 ETF", AssetType.ETF, "NYSE ARCA", description="S&P 500 Index"),
    SymbolInfo("ARKK", "ARK Innovation ETF", AssetType.ETF, "NYSE ARCA", description="Disruptive Innovation"),
    SymbolInfo("VWO", "Vanguard FTSE Emerging Markets ETF", AssetType.ETF, "NYSE ARCA", description="Emerging Markets"),
    SymbolInfo("BND", "Vanguard Total Bond Market ETF", AssetType.ETF, "NASDAQ", description="US Bond Market"),
    SymbolInfo("EFA", "iShares MSCI EAFE ETF", AssetType.ETF, "NYSE ARCA", description="Developed Markets ex-US"),
    SymbolInfo("IWM", "iShares Russell 2000 ETF", AssetType.ETF, "NYSE ARCA", description="Small-Cap US Stocks"),
    SymbolInfo("GLD", "SPDR Gold Shares", AssetType.ETF, "NYSE ARCA", description="Gold Bullion"),
]

# =============================================================================
# FUTURES - Commodity and Index Futures
# =============================================================================
FUTURES: list[SymbolInfo] = [
    SymbolInfo("/ES", "E-mini S&P 500", AssetType.FUTURES, "CME", description="S&P 500 Index Futures"),
    SymbolInfo("/NQ", "E-mini NASDAQ-100", AssetType.FUTURES, "CME", description="NASDAQ-100 Index Futures"),
    SymbolInfo("/YM", "E-mini Dow", AssetType.FUTURES, "CBOT", description="Dow Jones Index Futures"),
    SymbolInfo("/GC", "Gold Futures", AssetType.FUTURES, "COMEX", description="Gold Commodity Futures"),
    SymbolInfo("/CL", "Crude Oil Futures", AssetType.FUTURES, "NYMEX", description="WTI Crude Oil Futures"),
    SymbolInfo("/ZB", "30-Year Treasury Bond", AssetType.FUTURES, "CBOT", description="US Treasury Bond Futures"),
    SymbolInfo("/ZC", "Corn Futures", AssetType.FUTURES, "CBOT", description="Corn Commodity Futures"),
    SymbolInfo("/ZS", "Soybean Futures", AssetType.FUTURES, "CBOT", description="Soybean Commodity Futures"),
    SymbolInfo("/ZW", "Wheat Futures", AssetType.FUTURES, "CBOT", description="Wheat Commodity Futures"),
    SymbolInfo("/HG", "Copper Futures", AssetType.FUTURES, "COMEX", description="Copper Commodity Futures"),
]

# =============================================================================
# OPTIONS - Major Options Chains (represented as underlying + type)
# =============================================================================
OPTIONS: list[SymbolInfo] = [
    SymbolInfo("AAPL", "Apple Options", AssetType.OPTIONS, "CBOE", description="AAPL calls/puts"),
    SymbolInfo("SPY", "SPY Options", AssetType.OPTIONS, "CBOE", description="SPY calls/puts"),
    SymbolInfo("QQQ", "QQQ Options", AssetType.OPTIONS, "CBOE", description="QQQ calls/puts"),
    SymbolInfo("TSLA", "Tesla Options", AssetType.OPTIONS, "CBOE", description="TSLA calls/puts"),
    SymbolInfo("AMZN", "Amazon Options", AssetType.OPTIONS, "CBOE", description="AMZN calls/puts"),
]

# =============================================================================
# FOREX - Major Currency Pairs
# =============================================================================
CFDS: list[SymbolInfo] = [
    SymbolInfo("GC=F", "Gold (XAUUSD)", AssetType.FUTURES, "COMEX", "USD", description="Gold Spot / US Dollar"),
    SymbolInfo("SI=F", "Silver (XAGUSD)", AssetType.FUTURES, "COMEX", "USD", description="Silver Spot / US Dollar"),
    SymbolInfo("CL=F", "Crude Oil WTI", AssetType.FUTURES, "NYMEX", "USD", description="WTI Crude Oil"),
    SymbolInfo("HG=F", "Copper", AssetType.FUTURES, "COMEX", "USD", description="Copper Futures"),
    SymbolInfo("BZ=F", "Brent Crude Oil", AssetType.FUTURES, "NYMEX", "USD", description="Brent Crude Oil"),
    SymbolInfo("NG=F", "Natural Gas", AssetType.FUTURES, "NYMEX", "USD", description="Natural Gas Futures"),
    SymbolInfo("PL=F", "Platinum", AssetType.FUTURES, "COMEX", "USD", description="Platinum Futures"),
    SymbolInfo("PA=F", "Palladium", AssetType.FUTURES, "COMEX", "USD", description="Palladium Futures"),
]

FOREX: list[SymbolInfo] = [
    SymbolInfo("EURUSD=X", "EUR/USD", AssetType.FOREX, "FOREX", "USD", description="Euro / US Dollar"),
    SymbolInfo("GBPUSD=X", "GBP/USD", AssetType.FOREX, "FOREX", "USD", description="British Pound / US Dollar"),
    SymbolInfo("USDJPY=X", "USD/JPY", AssetType.FOREX, "FOREX", "JPY", description="US Dollar / Japanese Yen"),
    SymbolInfo("AUDUSD=X", "AUD/USD", AssetType.FOREX, "FOREX", "USD", description="Australian Dollar / US Dollar"),
    SymbolInfo("USDCAD=X", "USD/CAD", AssetType.FOREX, "FOREX", "CAD", description="US Dollar / Canadian Dollar"),
    SymbolInfo("NZDUSD=X", "NZD/USD", AssetType.FOREX, "FOREX", "USD", description="New Zealand Dollar / US Dollar"),
    SymbolInfo("USDCHF=X", "USD/CHF", AssetType.FOREX, "FOREX", "CHF", description="US Dollar / Swiss Franc"),
    SymbolInfo("EURGBP=X", "EUR/GBP", AssetType.FOREX, "FOREX", "GBP", description="Euro / British Pound"),
    SymbolInfo("EURJPY=X", "EUR/JPY", AssetType.FOREX, "FOREX", "JPY", description="Euro / Japanese Yen"),
    SymbolInfo("GBPJPY=X", "GBP/JPY", AssetType.FOREX, "FOREX", "JPY", description="British Pound / Japanese Yen"),
]

# =============================================================================
# CRYPTO - Cryptocurrencies
# =============================================================================
CRYPTO: list[SymbolInfo] = [
    SymbolInfo("BTC-USD", "Bitcoin", AssetType.CRYPTO, "CRYPTO", description="Bitcoin / US Dollar"),
    SymbolInfo("ETH-USD", "Ethereum", AssetType.CRYPTO, "CRYPTO", description="Ethereum / US Dollar"),
    SymbolInfo("SOL-USD", "Solana", AssetType.CRYPTO, "CRYPTO", description="Solana / US Dollar"),
    SymbolInfo("ADA-USD", "Cardano", AssetType.CRYPTO, "CRYPTO", description="Cardano / US Dollar"),
    SymbolInfo("XRP-USD", "Ripple", AssetType.CRYPTO, "CRYPTO", description="XRP / US Dollar"),
    SymbolInfo("BNB-USD", "Binance Coin", AssetType.CRYPTO, "CRYPTO", description="Binance Coin / US Dollar"),
    SymbolInfo("DOT-USD", "Polkadot", AssetType.CRYPTO, "CRYPTO", description="Polkadot / US Dollar"),
    SymbolInfo("LINK-USD", "Chainlink", AssetType.CRYPTO, "CRYPTO", description="Chainlink / US Dollar"),
    SymbolInfo("DOGE-USD", "Dogecoin", AssetType.CRYPTO, "CRYPTO", description="Dogecoin / US Dollar"),
    SymbolInfo("LTC-USD", "Litecoin", AssetType.CRYPTO, "CRYPTO", description="Litecoin / US Dollar"),
]

# =============================================================================
# BONDS - Government Bond Yields
# =============================================================================
BONDS: list[SymbolInfo] = [
    SymbolInfo("^TNX", "US 10-Year Treasury", AssetType.BOND, "INDEX", "USD", description="US 10-Year Treasury Yield"),
    SymbolInfo("^IRX", "US 2-Year Treasury", AssetType.BOND, "INDEX", "USD", description="US 2-Year Treasury Yield"),
    SymbolInfo("^TYX", "US 30-Year Treasury", AssetType.BOND, "INDEX", "USD", description="US 30-Year Treasury Yield"),
    SymbolInfo("DE10Y.DE", "Germany 10-Year Bond", AssetType.BOND, "XETRA", "EUR", description="German 10-Year Bund"),
    SymbolInfo("GB10Y.L", "UK 10-Year Gilt", AssetType.BOND, "LSE", "GBP", description="UK 10-Year Gilt"),
    SymbolInfo("JP10Y.T", "Japan 10-Year Bond", AssetType.BOND, "TSE", "JPY", description="Japan 10-Year JGB"),
    SymbolInfo("CN10Y.SS", "China 10-Year Bond", AssetType.BOND, "SSE", "CNY", description="China 10-Year Government Bond"),
    SymbolInfo("AU10Y.AX", "Australia 10-Year Bond", AssetType.BOND, "ASX", "AUD", description="Australia 10-Year Government Bond"),
    SymbolInfo("CA10Y.TO", "Canada 10-Year Bond", AssetType.BOND, "TSX", "CAD", description="Canada 10-Year Government Bond"),
    SymbolInfo("IT10Y.MI", "Italy 10-Year Bond", AssetType.BOND, "MIL", "EUR", description="Italy 10-Year BTP"),
]


# =============================================================================
# UNIVERSE - Combined symbol registry
# =============================================================================
def get_all_symbols() -> list[SymbolInfo]:
    """Return all symbols across all asset types."""
    return STOCKS + FUNDS + FUTURES + CFDS + OPTIONS + FOREX + CRYPTO + BONDS


def get_symbols_by_type(asset_type: AssetType) -> list[SymbolInfo]:
    """Return symbols filtered by asset type."""
    mapping = {
        AssetType.STOCK: STOCKS,
        AssetType.ETF: FUNDS,
        AssetType.FUTURES: FUTURES,
        AssetType.OPTIONS: OPTIONS,
        AssetType.FOREX: FOREX,
        AssetType.CRYPTO: CRYPTO,
        AssetType.BOND: BONDS,
    }
    return mapping.get(asset_type, [])


def get_symbol_info(symbol: str) -> Optional[SymbolInfo]:
    """Look up symbol metadata by ticker."""
    for info in get_all_symbols():
        if info.symbol == symbol:
            return info
    return None


def search_symbols(query: str, asset_type: Optional[AssetType] = None) -> list[SymbolInfo]:
    """Search symbols by name or ticker."""
    query = query.upper()
    symbols = get_symbols_by_type(asset_type) if asset_type else get_all_symbols()
    return [
        s for s in symbols
        if query in s.symbol.upper() or query in s.name.upper()
    ]


# Default symbol for UI
DEFAULT_SYMBOL = "AAPL"

# Available timeframes
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]
DEFAULT_TIMEFRAME = "1d"
