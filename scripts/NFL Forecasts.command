#!/bin/bash
# Double-click this file to open the NFL Forecasting Lab dashboard in your browser.
# A Terminal window opens and must stay open while you use the dashboard.
# Close that window when you're done.
PROJECT="$HOME/code/forecast_lab"
cd "$PROJECT" 2>/dev/null || { echo "Can't find the project at $PROJECT"; read -n 1 -p "Press any key to close."; exit 1; }
if ! .venv/bin/forecast-lab dashboard --help >/dev/null 2>&1; then
  echo "The dashboard isn't installed in this copy yet."
  echo "Merge the dashboard pull request on GitHub, then run:  git -C ~/code/forecast_lab pull"
  read -n 1 -p "Press any key to close."; exit 1
fi
echo "Opening the NFL Forecasting Lab dashboard..."
exec .venv/bin/forecast-lab dashboard
