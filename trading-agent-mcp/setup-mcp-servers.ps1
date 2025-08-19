# ========================================
# MCP SERVERS AUTOMATED SETUP SCRIPT
# ========================================
# PowerShell script to install and configure MCP servers for trading

Write-Host "🚀 Starting MCP Trading Servers Setup..." -ForegroundColor Green

# Create main directory if it doesn't exist
if (!(Test-Path "trading-agent-mcp")) {
    mkdir trading-agent-mcp
    Write-Host "✅ Created trading-agent-mcp directory" -ForegroundColor Green
}

cd trading-agent-mcp

# ========================================
# INSTALL MCP SERVERS
# ========================================
Write-Host "📦 Installing MCP servers..." -ForegroundColor Yellow

# Trading servers
Write-Host "Installing Alpaca trading server..." -ForegroundColor Cyan
git clone https://github.com/alpacahq/alpaca-mcp-server.git

Write-Host "Installing crypto price server..." -ForegroundColor Cyan
git clone https://github.com/truss44/mcp-crypto-price.git

# Development servers  
Write-Host "Installing GitHub MCP server..." -ForegroundColor Cyan
git clone https://github.com/github/github-mcp-server.git

Write-Host "Installing official MCP servers..." -ForegroundColor Cyan
git clone https://github.com/modelcontextprotocol/servers.git

# Data servers
Write-Host "Installing Playwright web automation..." -ForegroundColor Cyan
git clone https://github.com/microsoft/playwright-mcp.git

# ========================================
# INSTALL DEPENDENCIES
# ========================================
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow

# Alpaca (Python)
Write-Host "Installing Alpaca dependencies..." -ForegroundColor Cyan
cd alpaca-mcp-server
pip install -r requirements.txt
cd ..

# Crypto Price (Node.js)
Write-Host "Installing crypto price dependencies..." -ForegroundColor Cyan
cd mcp-crypto-price
npm install
cd ..

# Playwright (Node.js)
Write-Host "Installing Playwright dependencies..." -ForegroundColor Cyan
cd playwright-mcp
npm install
cd ..

# Filesystem (Node.js)
Write-Host "Installing filesystem server dependencies..." -ForegroundColor Cyan
cd servers/src/filesystem
npm install
cd ../../..

# GitHub (Go - requires Go to be installed)
Write-Host "Attempting to build GitHub server (requires Go)..." -ForegroundColor Cyan
cd github-mcp-server
if (Get-Command go -ErrorAction SilentlyContinue) {
    go build
    Write-Host "✅ GitHub server built successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️  Go not found - install Go to use GitHub server" -ForegroundColor Red
    Write-Host "   Download from: https://golang.org/dl/" -ForegroundColor Yellow
}
cd ..

# ========================================
# CREATE CONFIGURATION FILES
# ========================================
Write-Host "⚙️  Creating configuration files..." -ForegroundColor Yellow

# Create .cursor directory if it doesn't exist
if (!(Test-Path ".cursor")) {
    mkdir .cursor
}

# Create MCP configuration for Cursor
$mcpConfig = @'
{
  "mcpServers": {
    "alpaca-trading": {
      "type": "stdio",
      "command": "python",
      "args": ["alpaca-mcp-server/alpaca_mcp_server.py"],
      "env": {
        "ALPACA_API_KEY": "${ALPACA_API_KEY}",
        "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}",
        "ALPACA_PAPER_TRADE": "True"
      }
    },
    "crypto-price": {
      "type": "stdio", 
      "command": "node",
      "args": ["mcp-crypto-price/dist/index.js"]
    },
    "filesystem": {
      "type": "stdio",
      "command": "node", 
      "args": ["servers/src/filesystem/dist/index.js"],
      "env": {
        "ALLOWED_DIRECTORIES": "C:\Users\PC\Desktop\Agent-trader"
      }
    },
    "playwright": {
      "type": "stdio",
      "command": "node",
      "args": ["playwright-mcp/lib/browserServer.js"]
    }
  }
}
'@

$mcpConfig | Out-File -FilePath ".cursor/mcp-config.json" -Encoding UTF8
Write-Host "✅ Created Cursor MCP configuration" -ForegroundColor Green

# ========================================
# COMPLETION MESSAGE
# ========================================
Write-Host "🎉 MCP Servers Installation Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Copy env-template.txt to .env and add your API keys" -ForegroundColor White
Write-Host "2. Install Go (https://golang.org/dl/) to use GitHub server" -ForegroundColor White
Write-Host "3. Restart Cursor to load the new MCP servers" -ForegroundColor White
Write-Host "4. Test servers with natural language queries" -ForegroundColor White
Write-Host ""
Write-Host "Available Trading Tools:" -ForegroundColor Green
Write-Host "🔹 Alpaca: Stock/options trading, portfolio management" -ForegroundColor Cyan
Write-Host "🔹 Crypto Price: Real-time crypto data and analysis" -ForegroundColor Cyan
Write-Host "🔹 Playwright: Web automation and data scraping" -ForegroundColor Cyan
Write-Host "🔹 Filesystem: Secure file operations" -ForegroundColor Cyan
Write-Host "🔹 GitHub: Repository management (when Go is installed)" -ForegroundColor Cyan
