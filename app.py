import os, re, sqlite3
from datetime import date

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, g
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-for-real-use")
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bills.db")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------- database ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    conn = g.pop("db", None)
    if conn:
        conn.close()


def init_db():
    with sqlite3.connect(DB) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS bills(month TEXT PRIMARY KEY, units REAL NOT NULL, amount REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);
        """)


# ---------- helpers ----------
@app.template_filter("inr")
def inr(v):
    return "₹" + re.sub(r"(\d)(?=(\d\d)+\d$)", r"\1,", str(int(round(v))))


@app.template_filter("label")
def label(m):
    y, mo = m.split("-")
    return f"{MONTHS[int(mo) - 1]} {y}"


def midx(m):
    y, mo = m.split("-")
    return int(y) * 12 + int(mo) - 1


def mkey(i):
    return f"{i // 12}-{i % 12 + 1:02d}"


def user_bills():
    return [dict(r) for r in db().execute("SELECT month, units, amount FROM bills ORDER BY month")]


def predict(rows, field):
    """Next month's value. Linear regression (scikit-learn) if >= 3 months, else average."""
    if not rows:
        return None
    rows = rows[-12:]
    ys = [r[field] for r in rows]
    nxt = midx(rows[-1]["month"]) + 1
    if len(rows) < 3:
        return {"value": sum(ys) / len(ys), "method": "average", "slope": 0, "r2": None, "next": mkey(nxt)}
    X = np.array([[midx(r["month"])] for r in rows])
    model = LinearRegression().fit(X, ys)
    return {"value": max(0.0, float(model.predict([[nxt]])[0])), "method": "linear regression",
            "slope": float(model.coef_[0]), "r2": float(model.score(X, ys)), "next": mkey(nxt)}


def suggestions(rows, pa):
    n, t = len(rows), []
    if n < 3:
        t.append("Add at least 3 months of bills so the prediction can use a trend.")
    if n:
        last, avg = rows[-1], sum(r["amount"] for r in rows) / n
        if n > 1 and rows[-2]["amount"] > 0:
            d = (last["amount"] - rows[-2]["amount"]) / rows[-2]["amount"] * 100
            if d > 10:
                t.append(f"Your {label(last['month'])} bill is {d:.0f}% higher than {label(rows[-2]['month'])}. "
                         "Check heavy appliances such as the AC, geyser and water pump.")
            elif d < -5:
                t.append(f"Your bill dropped {abs(d):.0f}% compared with {label(rows[-2]['month'])}. Whatever you changed is working.")
        if last["amount"] > avg * 1.1:
            t.append(f"Your latest bill is above your average of {inr(avg)}. Compare that month's usage with a typical month.")
        tu = sum(r["units"] for r in rows)
        if tu and last["units"] and last["amount"] / last["units"] > sum(r["amount"] for r in rows) / tu * 1.05:
            t.append("Your cost per unit rose this month. Higher usage can push you into costlier slabs.")
        if pa["value"] > last["amount"] * 1.05:
            t.append(f"Next month is estimated to rise to {inr(pa['value'])}. Cutting units by 10% could save about {inr(pa['value'] * .1)}.")
        else:
            t.append(f"Next month looks steady at about {inr(pa['value'])}. Keep your current usage to stay near this.")
        mx = max(rows, key=lambda r: r["amount"])
        t.append(f"Your highest bill was {inr(mx['amount'])} in {label(mx['month'])}. Watch for the same conditions again.")
    return t


# ---------- pages ----------
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/dashboard")
def dashboard():
    rows = user_bills()
    stats = None
    if rows:
        stats = {"last": rows[-1], "prev": rows[-2] if len(rows) > 1 else None,
                 "units": sum(r["units"] for r in rows), "avg": sum(r["amount"] for r in rows) / len(rows),
                 "pred": predict(rows, "amount")}
        if stats["prev"] and stats["prev"]["amount"]:
            stats["change"] = (rows[-1]["amount"] - stats["prev"]["amount"]) / stats["prev"]["amount"] * 100
    b = db().execute("SELECT v FROM settings WHERE k='budget'").fetchone()
    return render_template("dashboard.html", rows=rows, s=stats, budget=float(b["v"]) if b and b["v"] else 0)


@app.route("/budget", methods=["POST"])
def budget():
    try:
        v = max(0.0, float(request.form["budget"]))
    except (ValueError, KeyError):
        v = 0.0
    db().execute("INSERT INTO settings(k,v) VALUES('budget',?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (str(v),))
    db().commit()
    flash("Budget saved." if v else "Budget cleared.")
    return redirect(url_for("dashboard"))


@app.route("/add", methods=["GET", "POST"])
def add():
    form = {"month": request.args.get("month", date.today().strftime("%Y-%m")), "units": "", "amount": ""}
    if request.method == "GET" and request.args.get("month"):
        r = db().execute("SELECT * FROM bills WHERE month=?", (form["month"],)).fetchone()
        if r:
            form.update(units=r["units"], amount=r["amount"])
    if request.method == "POST":
        try:
            m = request.form["month"]
            u, a = float(request.form["units"]), float(request.form["amount"])
            if not re.fullmatch(r"\d{4}-\d{2}", m) or u < 0 or a < 0:
                raise ValueError
        except (ValueError, KeyError):
            flash("Enter a month, units and amount. Numbers must be zero or more.")
            return render_template("add.html", form=request.form)
        db().execute("""INSERT INTO bills(month,units,amount) VALUES(?,?,?)
                        ON CONFLICT(month) DO UPDATE SET units=excluded.units, amount=excluded.amount""", (m, u, a))
        db().commit()
        flash(f"Saved the {label(m)} bill.")
        return redirect(url_for("history"))
    return render_template("add.html", form=form)


@app.route("/history")
def history():
    return render_template("history.html", rows=list(reversed(user_bills())))


@app.route("/delete/<month>", methods=["POST"])
def delete(month):
    db().execute("DELETE FROM bills WHERE month=?", (month,))
    db().commit()
    flash(f"Deleted the {label(month)} bill.")
    return redirect(url_for("history"))


@app.route("/predict")
def predict_page():
    rows = user_bills()
    if not rows:
        return render_template("predict.html", rows=rows)
    pa, pu = predict(rows, "amount"), predict(rows, "units")
    return render_template("predict.html", rows=rows, pa=pa, pu=pu, tips=suggestions(rows, pa),
                           mx=max(rows, key=lambda r: r["amount"]))


@app.route("/sample", methods=["POST"])
def sample():
    now = date.today().year * 12 + date.today().month - 1
    for i, u in enumerate([236, 248, 271, 305, 284, 262]):
        db().execute("INSERT OR IGNORE INTO bills(month,units,amount) VALUES(?,?,?)",
                     (mkey(now - 6 + i), u, round(u * (5.5 + u / 350))))
    db().commit()
    flash("Added 6 months of sample bills. You can edit or delete them any time.")
    return redirect(url_for("dashboard"))


init_db()
if __name__ == "__main__":
    app.run(host="0.0.0.0")
