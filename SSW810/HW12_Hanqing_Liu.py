from flask import Flask, render_template
import sqlite3

app = Flask(__name__)

@app.route("/hello")
def hello_world():
    return "Hello World"

@app.route("/goodbye")
def goodbye():
    return "Goodbye"

@app.route("/template")
def render_temp():
    return render_template("param.html", my_header="I am", my_param="Who I am")

@app.route("/Stevens_Repository")
def steve_repo():
    try:
        db_url = "/Users/hanqingliu/Desktop/SSW810-HW12/810_startup.db"
        db = sqlite3.connect(db_url)
    except sqlite3.OperationalError:
        return f"Could not find the db with the url {db_url}"
    else:
        query = "select i.CWID, i.Name, i.Dept, g.Course, count(g.Course) as Students from grades g join instructors i on i.CWID=g.InstructorCWID group by i.Name, g.Course"
        data = [{"cwid":cwid, "name":name, "dept":dept, "course":course, "student":student} for cwid, name, dept, course, student in db.execute(query)]
        db.close()
        '''flask has a serious CSS cache problem that they need to deal with'''
        return render_template(
            "instructors.html",
            title="Stevens Repository",
            table_title="Number of Students by Course and Instructor",
            instructors=data
        )

app.run(debug=True)