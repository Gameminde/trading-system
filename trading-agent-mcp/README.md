# 🚀 Trading Agent MCP Servers

A comprehensive collection of Model Context Protocol (MCP) servers designed to enhance AI capabilities for algorithmic trading, cryptocurrency analysis, and financial data processing.

## 🎯 Overview

This repository provides ready-to-use MCP servers that enable AI assistants (like Claude, Cursor, and others) to:

- **Trade stocks and options** via Alpaca Markets
- **Analyze cryptocurrency** prices and market data
- **Automate web scraping** with Playwright
- **Manage files** securely with filesystem access
- **Interact with GitHub** repositories (requires Go)

## 📦 Installed Servers

### 🏦 Trading & Finance
- **Alpaca Trading** (`alpaca-mcp-server`) - Stock/options trading, portfolio management, real-time market data
- **Crypto Price** (`mcp-crypto-price`) - Cryptocurrency price tracking, market analysis, historical trends

### 🔧 Development & Automation  
- **GitHub MCP** (`github-mcp-server`) - Repository management, issue tracking, pull requests (Go required)
- **Playwright** (`playwright-mcp`) - Web automation, browser control, data scraping
- **Filesystem** (`servers/src/filesystem`) - Secure file operations, directory access control

## ⚡ Quick Start

### 1. Configure Environment Variables
```bash
# Copy template and add your API keys
copy env-template.txt .env
# Edit .env with your actual credentials
```

### 2. Install Go (Optional - for GitHub server)
Download and install Go from: https://golang.org/dl/
```bash
cd github-mcp-server
go build
```

### 3. Configure Cursor
The MCP configuration is already created at `.cursor/mcp-config.json`. Just restart Cursor to load the servers.

### 4. Test Your Setup
Try these natural language queries in Cursor:

```
"What's my current account balance on Alpaca?"
"Show me Bitcoin's price and 24h change"
"List files in my project directory"
"Open a web page and extract all the headlines"
```

## 🔑 Required API Keys

### Alpaca Trading
1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Generate API keys from your dashboard
3. Start with paper trading (ALPACA_PAPER_TRADE=True)

### GitHub (Optional)
1. Go to [GitHub Settings → Tokens](https://github.com/settings/tokens)
2. Create a personal access token
3. Add necessary permissions for your use case

## 🛠️ Manual Installation

If the automated script fails, run these commands manually:

```powershell
# Create directory
mkdir trading-agent-mcp
cd trading-agent-mcp

# Clone repositories
git clone https://github.com/alpacahq/alpaca-mcp-server.git
git clone https://github.com/truss44/mcp-crypto-price.git
git clone https://github.com/github/github-mcp-server.git
git clone https://github.com/modelcontextprotocol/servers.git
git clone https://github.com/microsoft/playwright-mcp.git

# Install dependencies
cd alpaca-mcp-server && pip install -r requirements.txt && cd ..
cd mcp-crypto-price && npm install && cd ..
cd playwright-mcp && npm install && cd ..
cd servers/src/filesystem && npm install && cd ../../..
```

## 📂 Directory Structure

```
trading-agent-mcp/
├── .cursor/
│   └── mcp-config.json          # Cursor MCP configuration
├── alpaca-mcp-server/           # Stock/options trading
├── mcp-crypto-price/           # Cryptocurrency data
├── github-mcp-server/          # GitHub integration (Go)
├── playwright-mcp/             # Web automation
├── servers/src/filesystem/     # File system access
├── env-template.txt            # Environment variables template
├── setup-mcp-servers.ps1       # Automated installation script
└── README.md                   # This file
```

## 🔐 Security Notes

- **Never commit API keys** to version control
- **Use paper trading** initially (ALPACA_PAPER_TRADE=True)
- **Limit filesystem access** to specific directories
- **Review all trades** before execution in live accounts

## 🎯 Trading Capabilities

### Alpaca Trading Server
- View account balance and positions
- Place stock and options orders
- Get real-time market data
- Manage watchlists
- Access corporate actions and earnings data
- Execute complex options strategies

### Crypto Price Server
- Real-time cryptocurrency prices
- Market cap and volume data
- Historical price trends
- Price change analysis
- Multi-currency support

## 🌐 Web Automation

### Playwright Server
- Automated web browsing
- Data extraction from websites
- Screenshot capture
- Form automation
- JavaScript execution

## 🚨 Troubleshooting

### Common Issues

1. **Server not found**: Ensure all dependencies are installed
2. **Permission denied**: Check file/directory permissions
3. **API authentication**: Verify your API keys are correct
4. **Go not installed**: GitHub server requires Go runtime

### Debug Mode
Add environment variable `DEBUG=1` for verbose logging.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project includes multiple open-source components:
- Alpaca MCP Server: MIT License
- Other servers: See individual LICENSE files

## 🔗 Related Resources

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Alpaca Markets Documentation](https://alpaca.markets/docs/)
- [Cursor MCP Guide](https://docs.cursor.com/context/mcp)
- [Playwright Documentation](https://playwright.dev/)

---

**⚠️ Risk Disclaimer**: Trading involves substantial risk. Always test with paper accounts before using real funds. Past performance doesn't guarantee future results.
