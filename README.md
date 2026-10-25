# Electricity Bill Tracker with Prediction

A Flask website where each user saves monthly electricity units and bills, sees charts, and gets an estimate of next month's bill.

## Run in VS Code (Windows)
1. Open this folder in VS Code and press Ctrl + ` to open the terminal.
2. `pip install -r requirements.txt`  (if `pip` is not found: `py -m pip install -r requirements.txt`)
3. `python app.py`  (or `py app.py`)
4. Open http://127.0.0.1:5000 in the browser, go to Dashboard and click "Load sample data".

## Pages
Home, Dashboard (with monthly budget), Add bill, Bill history (edit and delete), Prediction and analytics.

## Technology
| Part | Technology |
|---|---|
| Pages and design | HTML, CSS, JavaScript (Jinja templates) |
| Backend | Python Flask |
| Database | SQLite (`bills.db`, created automatically) |
| Charts | Chart.js (loaded from a CDN, needs internet) |
| Prediction | scikit-learn LinearRegression |

## How the prediction works
The last 12 months are used. Each month is a number x (year*12 + month) and the bill is y. A straight line y = m*x + c is fitted by least squares, and the next month's x gives the predicted bill. With fewer than 3 months the site uses the average instead. R² on the Prediction page shows how well the line fits (closer to 1 is better). There is no login: it is a single-user app and all bills are stored in the local SQLite file.

## Files
`app.py` routes, database and prediction · `templates/` pages · `static/style.css` design · `requirements.txt` packages
