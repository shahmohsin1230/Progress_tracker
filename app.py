import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import json
import time
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Initialize session state variables if they don't exist
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

if 'team_members' not in st.session_state:
    st.session_state.team_members = []

if 'projects' not in st.session_state:
    st.session_state.projects = []

if 'weekly_reports' not in st.session_state:
    st.session_state.weekly_reports = []

if 'bottleneck_alerts' not in st.session_state:
    st.session_state.bottleneck_alerts = []

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

if 'slack_webhook' not in st.session_state:
    st.session_state.slack_webhook = ""

if 'email_settings' not in st.session_state:
    st.session_state.email_settings = {
        "smtp_server": "",
        "smtp_port": 587,
        "username": "",
        "password": "",
        "recipients": []
    }

def save_data():
    # Save tasks
    with open('tasks.json', 'w') as f:
        json.dump(st.session_state.tasks, f, indent=4)
    
    # Save team members
    with open('team_members.json', 'w') as f:
        json.dump(st.session_state.team_members, f, indent=4)
    
    # Save projects
    with open('projects.json', 'w') as f:
        json.dump(st.session_state.projects, f, indent=4)
    
    # Save weekly reports
    with open('weekly_reports.json', 'w') as f:
        json.dump(st.session_state.weekly_reports, f, indent=4)
    
    # Save bottleneck alerts
    with open('bottleneck_alerts.json', 'w') as f:
        json.dump(st.session_state.bottleneck_alerts, f, indent=4)
        # Save API settings
    settings = {
        "api_key": st.session_state.api_key,
        "slack_webhook": st.session_state.slack_webhook,
        "email_settings": st.session_state.email_settings
    }
    with open('settings.json', 'w') as f:
        json.dump(settings, f)


# Function to load data from files
def load_data():
    # Load tasks
    if os.path.exists('tasks.json'):
        with open('tasks.json', 'r') as f:
            st.session_state.tasks = json.load(f)
    
    # Load team members
    if os.path.exists('team_members.json'):
        with open('team_members.json', 'r') as f:
            st.session_state.team_members = json.load(f)
    
    # Load projects
    if os.path.exists('projects.json'):
        with open('projects.json', 'r') as f:
            st.session_state.projects = json.load(f)
    
    # Load weekly reports
    if os.path.exists('weekly_reports.json'):
        with open('weekly_reports.json', 'r') as f:
            st.session_state.weekly_reports = json.load(f)
    
    # Load bottleneck alerts
    if os.path.exists('bottleneck_alerts.json'):
        with open('bottleneck_alerts.json', 'r') as f:
            st.session_state.bottleneck_alerts = json.load(f)
    # Load API settings
    if os.path.exists('settings.json'):
        with open('settings.json', 'r') as f:
            settings = json.load(f)
            if 'api_key' in settings:
                st.session_state.api_key = settings['api_key']
            if 'slack_webhook' in settings:
                st.session_state.slack_webhook = settings['slack_webhook']
            if 'email_settings' in settings:
                st.session_state.email_settings = settings['email_settings']

# Load existing data when the app starts
load_data()

# Function to generate AI summary with GPT-4o
def generate_ai_summary(data, summary_type="weekly"):
    if not st.session_state.api_key:
        return "Please set your OpenAI API key in the Settings tab to generate AI summaries."
    
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        
        if summary_type == "weekly":
            prompt = f"""
            Generate a professional weekly summary report based on the following task and project data:
            
            {json.dumps(data, indent=2)}
            
            Include:
            1. Executive Summary
            2. Progress Overview 
            3. Completed Tasks
            4. In-Progress Tasks
            5. Delays and Blockers
            6. Team Performance Metrics
            7. Recommendations
            
            Format the report in a professional, concise manner suitable for executive review.
            """
        elif summary_type == "bottleneck":
            prompt = f"""
            Analyze the following task and project data to identify bottlenecks and potential issues:
            
            {json.dumps(data, indent=2)}
            
            Generate a bottleneck alert report that includes:
            1. Critical Issues Summary
            2. Specific Bottlenecks Identified
            3. Affected Projects/Teams
            4. Root Cause Analysis
            5. Recommended Actions
            6. Priority Level (High/Medium/Low)
            
            Format the report in a clear, actionable manner.
            """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a professional project management assistant that creates insightful reports based on project data."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating AI summary: {str(e)}"

# Function to send email
# Modify the send_email function to use the same HTML formatting as Slack
def send_email(subject, body):
    if not all([st.session_state.email_settings["smtp_server"], 
                st.session_state.email_settings["username"], 
                st.session_state.email_settings["password"], 
                st.session_state.email_settings["recipients"]]):
        return False, "Email settings are incomplete"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = st.session_state.email_settings["username"]
        msg['To'] = ", ".join(st.session_state.email_settings["recipients"])
        msg['Subject'] = subject
        
        # Format the body to preserve formatting similar to Slack
        # Convert markdown formatting to HTML
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                h1 {{ color: #333; }}
                h2 {{ color: #444; }}
                h3 {{ color: #555; }}
                .highlight {{ background-color: #f5f5f5; padding: 10px; border-left: 5px solid #2c87c5; }}
                ul {{ margin-left: 20px; }}
                li {{ margin-bottom: 5px; }}
            </style>
        </head>
        <body>
            {body.replace('\n', '<br>').replace('*', '<strong>').replace('_', '<em>')}
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        server = smtplib.SMTP(st.session_state.email_settings["smtp_server"], st.session_state.email_settings["smtp_port"])
        server.starttls()
        server.login(st.session_state.email_settings["username"], st.session_state.email_settings["password"])
        server.send_message(msg)
        server.quit()
        
        return True, "Email sent successfully"
    except Exception as e:
        return False, f"Error sending email: {str(e)}"

# Function to send Slack message
def send_slack_message(message):
    if not st.session_state.slack_webhook:
        return False, "Slack webhook URL is not set"
    
    try:
        response = requests.post(
            st.session_state.slack_webhook,
            json={"text": message},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return True, "Message sent to Slack successfully"
        else:
            return False, f"Error sending to Slack: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Error sending to Slack: {str(e)}"

# Function to calculate KPIs
def calculate_kpis():
    # Convert tasks to DataFrame for easier analysis
    if not st.session_state.tasks:
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "completion_rate": 0,
            "on_time_completion_rate": 0,
            "total_hours": 0,
            "team_performance": {},
            "project_status": {}
        }
    
    df = pd.DataFrame(st.session_state.tasks)
    
    # Basic KPIs
    total_tasks = len(df)
    completed_tasks = len(df[df['status'] == 'Completed'])
    completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
    
    # On-time completion
    df['deadline'] = pd.to_datetime(df['deadline'])
    df['completion_date'] = pd.to_datetime(df['completion_date'])
    on_time_tasks = 0
    for idx, task in df[df['status'] == 'Completed'].iterrows():
        if pd.notna(task['completion_date']) and pd.notna(task['deadline']):
            if task['completion_date'] <= task['deadline']:
                on_time_tasks += 1
    
    on_time_completion_rate = on_time_tasks / completed_tasks if completed_tasks > 0 else 0
    
    # Total hours spent
    total_hours = df['hours_spent'].sum()
    
    # Team performance
    team_performance = {}
    for member in st.session_state.team_members:
        member_tasks = df[df['assigned_to'] == member['name']]
        member_total = len(member_tasks)
        member_completed = len(member_tasks[member_tasks['status'] == 'Completed'])
        member_hours = member_tasks['hours_spent'].sum()
        
        if member_total > 0:
            member_completion_rate = member_completed / member_total
        else:
            member_completion_rate = 0
            
        on_time_member_tasks = 0
        for idx, task in member_tasks[member_tasks['status'] == 'Completed'].iterrows():
            if pd.notna(task['completion_date']) and pd.notna(task['deadline']):
                if task['completion_date'] <= task['deadline']:
                    on_time_member_tasks += 1
        
        member_on_time_rate = on_time_member_tasks / member_completed if member_completed > 0 else 0
        
        team_performance[member['name']] = {
            "total_tasks": member_total,
            "completed_tasks": member_completed,
            "completion_rate": member_completion_rate,
            "on_time_rate": member_on_time_rate,
            "hours_spent": member_hours
        }
    
    # Project status
    project_status = {}
    for project in st.session_state.projects:
        project_tasks = df[df['project'] == project['name']]
        project_total = len(project_tasks)
        project_completed = len(project_tasks[project_tasks['status'] == 'Completed'])
        project_hours = project_tasks['hours_spent'].sum()
        
        if project_total > 0:
            project_completion_rate = project_completed / project_total
        else:
            project_completion_rate = 0
            
        project_status[project['name']] = {
            "total_tasks": project_total,
            "completed_tasks": project_completed,
            "completion_rate": project_completion_rate,
            "hours_spent": project_hours
        }
    
    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "completion_rate": completion_rate,
        "on_time_completion_rate": on_time_completion_rate,
        "total_hours": total_hours,
        "team_performance": team_performance,
        "project_status": project_status
    }

# Function to identify bottlenecks
def identify_bottlenecks():
    if not st.session_state.tasks:
        return []
    
    bottlenecks = []
    df = pd.DataFrame(st.session_state.tasks)
    
    # Overdue tasks
    if 'deadline' in df.columns:
        df['deadline'] = pd.to_datetime(df['deadline'])
        today = datetime.datetime.now()
        overdue_tasks = df[(df['status'] != 'Completed') & (df['deadline'] < today)]
        
        for _, task in overdue_tasks.iterrows():
            bottlenecks.append({
                "type": "overdue_task",
                "severity": "high",
                "task_id": task.get('id', ''),
                "task_name": task.get('name', ''),
                "assigned_to": task.get('assigned_to', ''),
                "project": task.get('project', ''),
                "deadline": task.get('deadline', ''),
                "days_overdue": (today - task['deadline']).days if pd.notna(task['deadline']) else 0
            })
    
    # Team member overload
    member_task_counts = df[df['status'] != 'Completed'].groupby('assigned_to').size()
    for member, count in member_task_counts.items():
        if count > 5:  # Arbitrary threshold
            bottlenecks.append({
                "type": "member_overload",
                "severity": "medium",
                "member": member,
                "open_tasks": int(count)
            })
    
    # Project delays
    for project in st.session_state.projects:
        project_tasks = df[df['project'] == project['name']]
        if len(project_tasks) > 0:
            completion_rate = len(project_tasks[project_tasks['status'] == 'Completed']) / len(project_tasks)
            if completion_rate < 0.3 and pd.to_datetime(project['deadline']) < datetime.datetime.now() + datetime.timedelta(days=7):
                bottlenecks.append({
                    "type": "project_delay",
                    "severity": "high",
                    "project": project['name'],
                    "completion_rate": completion_rate,
                    "deadline": project['deadline']
                })
    
    return bottlenecks

# Function to generate weekly report
def generate_weekly_report():
    # Get KPIs
    kpis = calculate_kpis()
    
    # Get bottlenecks
    bottlenecks = identify_bottlenecks()
    
    # Combine data for AI summary
    report_data = {
        "kpis": kpis,
        "bottlenecks": bottlenecks,
        "tasks": st.session_state.tasks,
        "team_members": st.session_state.team_members,
        "projects": st.session_state.projects,
        "report_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "report_period": f"{(datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')} to {datetime.datetime.now().strftime('%Y-%m-%d')}"
    }
    
    # Generate AI summary
    ai_summary = generate_ai_summary(report_data, "weekly")
    
    # Create report object
    report = {
        "id": len(st.session_state.weekly_reports) + 1,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "period": f"{(datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')} to {datetime.datetime.now().strftime('%Y-%m-%d')}",
        "kpis": kpis,
        "bottlenecks": bottlenecks,
        "ai_summary": ai_summary
    }
    
    # Add to reports list
    st.session_state.weekly_reports.append(report)
    
    # Save data
    save_data()
    
    return report

# Function to generate bottleneck alert
def generate_bottleneck_alert():
    # Get bottlenecks
    bottlenecks = identify_bottlenecks()
    
    if not bottlenecks:
        return None
    
    # Combine data for AI summary
    alert_data = {
        "bottlenecks": bottlenecks,
        "tasks": st.session_state.tasks,
        "team_members": st.session_state.team_members,
        "projects": st.session_state.projects,
        "alert_date": datetime.datetime.now().strftime("%Y-%m-%d")
    }
    
    # Generate AI summary
    ai_summary = generate_ai_summary(alert_data, "bottleneck")
    
    # Create alert object
    alert = {
        "id": len(st.session_state.bottleneck_alerts) + 1,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "bottlenecks": bottlenecks,
        "ai_summary": ai_summary
    }
    
    # Add to alerts list
    st.session_state.bottleneck_alerts.append(alert)
    
    # Save data
    save_data()
    
    return alert

# Streamlit App Interface
st.set_page_config(page_title="AI KPI & Progress Tracker", layout="wide")

st.title("AI KPI & Progress Tracker")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Tasks", "Team", "Projects", "Reports", "Settings"])

# Dashboard page
if page == "Dashboard":
    st.header("KPI Dashboard")
    
    # Get KPIs
    kpis = calculate_kpis()
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tasks", kpis["total_tasks"])
    
    with col2:
        st.metric("Completed Tasks", kpis["completed_tasks"])
    
    with col3:
        st.metric("Completion Rate", f"{kpis['completion_rate']:.1%}")
    
    with col4:
        st.metric("On-Time Rate", f"{kpis['on_time_completion_rate']:.1%}")
    
    # Add more visualizations
    st.subheader("Task Status")
    if st.session_state.tasks:
        df = pd.DataFrame(st.session_state.tasks)
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig = px.pie(status_counts, values='Count', names='Status', title='Task Status Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tasks available. Add tasks to see analytics.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team Performance")
        if kpis["team_performance"]:
            team_df = pd.DataFrame.from_dict(kpis["team_performance"], orient='index')
            team_df['Member'] = team_df.index
            
            fig = px.bar(team_df, x='Member', y='completion_rate', title='Task Completion Rate by Team Member',
                        labels={'completion_rate': 'Completion Rate', 'Member': 'Team Member'})
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No team members or task data available.")
    
    with col2:
        st.subheader("Project Status")
        if kpis["project_status"]:
            project_df = pd.DataFrame.from_dict(kpis["project_status"], orient='index')
            project_df['Project'] = project_df.index
            
            fig = px.bar(project_df, x='Project', y='completion_rate', title='Project Completion Rate',
                        labels={'completion_rate': 'Completion Rate', 'Project': 'Project Name'})
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No projects or task data available.")
    
    # Bottlenecks
    st.subheader("Potential Bottlenecks")
    bottlenecks = identify_bottlenecks()
    
    if bottlenecks:
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "overdue_task":
                st.warning(f"⚠️ Overdue Task: '{bottleneck['task_name']}' assigned to {bottleneck['assigned_to']} is {bottleneck['days_overdue']} days overdue.")
            elif bottleneck["type"] == "member_overload":
                st.warning(f"⚠️ Team Member Overload: {bottleneck['member']} has {bottleneck['open_tasks']} open tasks.")
            elif bottleneck["type"] == "project_delay":
                st.warning(f"⚠️ Project Delay: '{bottleneck['project']}' is at only {bottleneck['completion_rate']:.1%} completion with deadline on {bottleneck['deadline']}.")
    else:
        st.success("No bottlenecks detected at this time.")
    
    # Weekly Report Generation
    st.subheader("Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Weekly Summary", key="gen_weekly"):
            with st.spinner("Generating weekly summary..."):
                report = generate_weekly_report()
                st.success(f"Weekly report #{report['id']} generated!")
    
    with col2:
        if st.button("Generate Bottleneck Alert", key="gen_bottleneck"):
            with st.spinner("Analyzing bottlenecks..."):
                alert = generate_bottleneck_alert()
                if alert:
                    st.success(f"Bottleneck alert #{alert['id']} generated!")
                else:
                    st.info("No significant bottlenecks detected.")
    
    # Distribution options
    st.subheader("Distribute Latest Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Send to Email", key="send_email"):
            if st.session_state.weekly_reports:
                latest_report = st.session_state.weekly_reports[-1]
                success, message = send_email(
                    f"Weekly Project Summary - {latest_report['date']}", 
                    latest_report['ai_summary']
                )
                if success:
                    st.success("Weekly report sent via email!")
                else:
                    st.error(message)
            else:
                st.error("No reports available to send. Generate a report first.")
    
    with col2:
        if st.button("Send to Slack", key="send_slack"):
            if st.session_state.weekly_reports:
                latest_report = st.session_state.weekly_reports[-1]
                success, message = send_slack_message(
                    f"*Weekly Project Summary - {latest_report['date']}*\n\n{latest_report['ai_summary']}"
                )
                if success:
                    st.success("Weekly report sent to Slack!")
                else:
                    st.error(message)
            else:
                st.error("No reports available to send. Generate a report first.")

# Tasks page
elif page == "Tasks":
    st.header("Task Management")
    
    # Task form
    with st.expander("Add New Task", expanded=False):
        with st.form("new_task_form"):
            task_name = st.text_input("Task Name")
            task_description = st.text_area("Description")
            
            col1, col2 = st.columns(2)
            
            with col1:
                project_options = [project["name"] for project in st.session_state.projects]
                task_project = st.selectbox("Project", [""] + project_options)
                
                task_status = st.selectbox("Status", ["Not Started", "In Progress", "Completed", "On Hold"])
                
                task_priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
            
            with col2:
                member_options = [member["name"] for member in st.session_state.team_members]
                task_assigned = st.selectbox("Assigned To", [""] + member_options)
                
                task_deadline = st.date_input("Deadline")
                
                task_hours = st.number_input("Hours Spent", min_value=0.0, step=0.5)
            
            task_completion_date = None
            if task_status == "Completed":
                task_completion_date = st.date_input("Completion Date")
            
            submitted = st.form_submit_button("Add Task")
            
            if submitted:
                if task_name and task_project and task_assigned:
                    # Create new task
                    new_task = {
                        "id": len(st.session_state.tasks) + 1,
                        "name": task_name,
                        "description": task_description,
                        "project": task_project,
                        "status": task_status,
                        "priority": task_priority,
                        "assigned_to": task_assigned,
                        "deadline": task_deadline.strftime("%Y-%m-%d"),
                        "hours_spent": task_hours,
                        "date_created": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "completion_date": task_completion_date.strftime("%Y-%m-%d") if task_completion_date else None
                    }
                    
                    st.session_state.tasks.append(new_task)
                    save_data()
                    st.success(f"Task '{task_name}' added successfully!")
                else:
                    st.error("Please fill out all required fields.")
    
    # Task list
    st.subheader("Task List")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect("Filter by Status", ["Not Started", "In Progress", "Completed", "On Hold"])
    
    with col2:
        project_options = [project["name"] for project in st.session_state.projects]
        project_filter = st.multiselect("Filter by Project", project_options)
    
    with col3:
        member_options = [member["name"] for member in st.session_state.team_members]
        member_filter = st.multiselect("Filter by Team Member", member_options)
    
    # Apply filters to tasks
    filtered_tasks = st.session_state.tasks
    
    if status_filter:
        filtered_tasks = [task for task in filtered_tasks if task["status"] in status_filter]
    
    if project_filter:
        filtered_tasks = [task for task in filtered_tasks if task["project"] in project_filter]
    
    if member_filter:
        filtered_tasks = [task for task in filtered_tasks if task["assigned_to"] in member_filter]
    
    # Convert to DataFrame for display
    if filtered_tasks:
        task_df = pd.DataFrame(filtered_tasks)
        # Reorder columns for better display
        display_columns = ['id', 'name', 'project', 'status', 'priority', 'assigned_to', 'deadline', 'hours_spent']
        display_df = task_df[display_columns]
        
        st.dataframe(display_df, hide_index=True)
        
        # Task editing
        st.subheader("Edit Task")
        
        task_to_edit = st.selectbox("Select Task to Edit", [f"{task['id']} - {task['name']}" for task in filtered_tasks])
        task_id = int(task_to_edit.split(" - ")[0])
        
        # Find the task
        task_index = next((i for i, task in enumerate(st.session_state.tasks) if task["id"] == task_id), None)
        
        if task_index is not None:
            task = st.session_state.tasks[task_index]
            
            with st.form("edit_task_form"):
                task_name = st.text_input("Task Name", value=task["name"])
                task_description = st.text_area("Description", value=task["description"])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    project_options = [project["name"] for project in st.session_state.projects]
                    task_project = st.selectbox("Project", [""] + project_options, index=0 if task["project"] not in project_options else project_options.index(task["project"]) + 1)
                    
                    status_options = ["Not Started", "In Progress", "Completed", "On Hold"]
                    task_status = st.selectbox("Status", status_options, index=status_options.index(task["status"]))
                    
                    priority_options = ["Low", "Medium", "High", "Critical"]
                    task_priority = st.selectbox("Priority", priority_options, index=priority_options.index(task["priority"]))
                
                with col2:
                    member_options = [member["name"] for member in st.session_state.team_members]
                    task_assigned = st.selectbox("Assigned To", [""] + member_options, index=0 if task["assigned_to"] not in member_options else member_options.index(task["assigned_to"]) + 1)
                    
                    task_deadline = st.date_input("Deadline", value=datetime.datetime.strptime(task["deadline"], "%Y-%m-%d") if task["deadline"] else None)
                    
                    task_hours = st.number_input("Hours Spent", min_value=0.0, step=0.5, value=float(task["hours_spent"]))
                
                task_completion_date = None
                if task_status == "Completed":
                    default_date = datetime.datetime.now()
                    if task["completion_date"]:
                        default_date = datetime.datetime.strptime(task["completion_date"], "%Y-%m-%d")
                    task_completion_date = st.date_input("Completion Date", value=default_date)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    submitted = st.form_submit_button("Update Task")
                
                with col2:
                    delete_button = st.form_submit_button("Delete Task", type="primary")
                
                if submitted:
                    if task_name and task_project and task_assigned:
                        # Update task
                        st.session_state.tasks[task_index].update({
                            "name": task_name,
                            "description": task_description,
                            "project": task_project,
                            "status": task_status,
                            "priority": task_priority,
                            "assigned_to": task_assigned,
                            "deadline": task_deadline.strftime("%Y-%m-%d"),
                            "hours_spent": task_hours,
                            "completion_date": task_completion_date.strftime("%Y-%m-%d") if task_completion_date else None
                        })
                        
                        save_data()
                        st.success(f"Task '{task_name}' updated successfully!")
                    else:
                        st.error("Please fill out all required fields.")
                
                if delete_button:
                    st.session_state.tasks.pop(task_index)
                    save_data()
                    st.success(f"Task deleted successfully!")
                    st.experimental_rerun()
    else:
        st.info("No tasks found with the selected filters. Add tasks or adjust your filters.")

# Team page (continued)
elif page == "Team":
    st.header("Team Management")
    
    # Team member form
    with st.expander("Add New Team Member", expanded=False):
        with st.form("new_team_member_form"):
            member_name = st.text_input("Name")
            member_role = st.text_input("Role")
            member_email = st.text_input("Email")
            member_skills = st.text_area("Skills (comma separated)")
            
            submitted = st.form_submit_button("Add Team Member")
            
            if submitted:
                if member_name and member_role:
                    # Create new team member
                    new_member = {
                        "id": len(st.session_state.team_members) + 1,
                        "name": member_name,
                        "role": member_role,
                        "email": member_email,
                        "skills": [skill.strip() for skill in member_skills.split(",") if skill.strip()],
                        "date_added": datetime.datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    st.session_state.team_members.append(new_member)
                    save_data()
                    st.success(f"Team member '{member_name}' added successfully!")
                else:
                    st.error("Please fill out all required fields.")
    
    # Team member list
    st.subheader("Team Members")
    
    if st.session_state.team_members:
        member_df = pd.DataFrame(st.session_state.team_members)
        # Exclude skills column for better display
        display_columns = ['id', 'name', 'role', 'email']
        display_df = member_df[display_columns] if all(col in member_df.columns for col in display_columns) else member_df
        
        st.dataframe(display_df, hide_index=True)
        
        # Member editing
        st.subheader("Edit Team Member")
        
        member_to_edit = st.selectbox("Select Team Member to Edit", 
                                     [f"{member['id']} - {member['name']}" for member in st.session_state.team_members])
        member_id = int(member_to_edit.split(" - ")[0])
        
        # Find the member
        member_index = next((i for i, member in enumerate(st.session_state.team_members) if member["id"] == member_id), None)
        
        if member_index is not None:
            member = st.session_state.team_members[member_index]
            
            with st.form("edit_member_form"):
                member_name = st.text_input("Name", value=member["name"])
                member_role = st.text_input("Role", value=member["role"])
                member_email = st.text_input("Email", value=member["email"])
                member_skills = st.text_area("Skills (comma separated)", 
                                           value=", ".join(member["skills"]) if "skills" in member else "")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    submitted = st.form_submit_button("Update Team Member")
                
                with col2:
                    delete_button = st.form_submit_button("Delete Team Member", type="primary")
                
                if submitted:
                    if member_name and member_role:
                        # Update member
                        st.session_state.team_members[member_index].update({
                            "name": member_name,
                            "role": member_role,
                            "email": member_email,
                            "skills": [skill.strip() for skill in member_skills.split(",") if skill.strip()]
                        })
                        
                        save_data()
                        st.success(f"Team member '{member_name}' updated successfully!")
                    else:
                        st.error("Please fill out all required fields.")
                
                if delete_button:
                    st.session_state.team_members.pop(member_index)
                    save_data()
                    st.success(f"Team member deleted successfully!")
                    st.experimental_rerun()
    else:
        st.info("No team members found. Add team members to get started.")
    
    # Workload visualization
    if st.session_state.team_members and st.session_state.tasks:
        st.subheader("Team Workload")
        
        # Calculate workload
        member_names = [member["name"] for member in st.session_state.team_members]
        tasks_df = pd.DataFrame(st.session_state.tasks)
        
        workload_data = []
        for member_name in member_names:
            member_tasks = tasks_df[tasks_df["assigned_to"] == member_name]
            open_tasks = len(member_tasks[member_tasks["status"] != "Completed"])
            hours_spent = member_tasks["hours_spent"].sum()
            
            workload_data.append({
                "Member": member_name,
                "Open Tasks": open_tasks,
                "Hours Spent": hours_spent
            })
        
        workload_df = pd.DataFrame(workload_data)
        
        # Plot workload
        fig = px.bar(workload_df, x="Member", y=["Open Tasks", "Hours Spent"], 
                    title="Team Workload Distribution",
                    barmode="group")
        st.plotly_chart(fig, use_container_width=True)

# Projects page
elif page == "Projects":
    st.header("Project Management")
    
    # Project form
    with st.expander("Add New Project", expanded=False):
        with st.form("new_project_form"):
            project_name = st.text_input("Project Name")
            project_description = st.text_area("Description")
            
            col1, col2 = st.columns(2)
            
            with col1:
                project_start_date = st.date_input("Start Date")
                project_status = st.selectbox("Status", ["Not Started", "In Progress", "Completed", "On Hold"])
            
            with col2:
                project_deadline = st.date_input("Deadline")
                project_priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
            
            submitted = st.form_submit_button("Add Project")
            
            if submitted:
                if project_name and project_description:
                    # Create new project
                    new_project = {
                        "id": len(st.session_state.projects) + 1,
                        "name": project_name,
                        "description": project_description,
                        "start_date": project_start_date.strftime("%Y-%m-%d"),
                        "deadline": project_deadline.strftime("%Y-%m-%d"),
                        "status": project_status,
                        "priority": project_priority,
                        "date_created": datetime.datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    st.session_state.projects.append(new_project)
                    save_data()
                    st.success(f"Project '{project_name}' added successfully!")
                else:
                    st.error("Please fill out all required fields.")
    
    # Project list
    st.subheader("Projects")
    
    if st.session_state.projects:
        project_df = pd.DataFrame(st.session_state.projects)
        display_columns = ['id', 'name', 'status', 'priority', 'start_date', 'deadline']
        display_df = project_df[display_columns] if all(col in project_df.columns for col in display_columns) else project_df
        
        st.dataframe(display_df, hide_index=True)
        
        # Project editing
        st.subheader("Edit Project")
        
        project_to_edit = st.selectbox("Select Project to Edit", 
                                      [f"{project['id']} - {project['name']}" for project in st.session_state.projects])
        project_id = int(project_to_edit.split(" - ")[0])
        
        # Find the project
        project_index = next((i for i, project in enumerate(st.session_state.projects) if project["id"] == project_id), None)
        
        if project_index is not None:
            project = st.session_state.projects[project_index]
            
            with st.form("edit_project_form"):
                project_name = st.text_input("Project Name", value=project["name"])
                project_description = st.text_area("Description", value=project["description"])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    project_start_date = st.date_input("Start Date", 
                                                     value=datetime.datetime.strptime(project["start_date"], "%Y-%m-%d"))
                    
                    status_options = ["Not Started", "In Progress", "Completed", "On Hold"]
                    project_status = st.selectbox("Status", status_options, 
                                                index=status_options.index(project["status"]))
                
                with col2:
                    project_deadline = st.date_input("Deadline", 
                                                   value=datetime.datetime.strptime(project["deadline"], "%Y-%m-%d"))
                    
                    priority_options = ["Low", "Medium", "High", "Critical"]
                    project_priority = st.selectbox("Priority", priority_options, 
                                                 index=priority_options.index(project["priority"]))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    submitted = st.form_submit_button("Update Project")
                
                with col2:
                    delete_button = st.form_submit_button("Delete Project", type="primary")
                
                if submitted:
                    if project_name and project_description:
                        # Update project
                        st.session_state.projects[project_index].update({
                            "name": project_name,
                            "description": project_description,
                            "start_date": project_start_date.strftime("%Y-%m-%d"),
                            "deadline": project_deadline.strftime("%Y-%m-%d"),
                            "status": project_status,
                            "priority": project_priority
                        })
                        
                        save_data()
                        st.success(f"Project '{project_name}' updated successfully!")
                    else:
                        st.error("Please fill out all required fields.")
                
                if delete_button:
                    st.session_state.projects.pop(project_index)
                    save_data()
                    st.success(f"Project deleted successfully!")
                    st.experimental_rerun()
    else:
        st.info("No projects found. Add projects to get started.")
    
    # Project details
    if st.session_state.projects and st.session_state.tasks:
        st.subheader("Project Details")
        
        project_to_view = st.selectbox("Select Project to View", 
                                      [project["name"] for project in st.session_state.projects],
                                      key="project_details")
        
        # Get project tasks
        tasks_df = pd.DataFrame(st.session_state.tasks)
        project_tasks = tasks_df[tasks_df["project"] == project_to_view]
        
        if not project_tasks.empty:
            # Task status breakdown
            st.write("Task Status Breakdown")
            status_counts = project_tasks["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            
            fig = px.pie(status_counts, values="Count", names="Status", 
                        title=f"Task Status for {project_to_view}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display tasks table
            st.write("Project Tasks")
            display_columns = ['id', 'name', 'status', 'priority', 'assigned_to', 'deadline', 'hours_spent']
            display_df = project_tasks[display_columns] if all(col in project_tasks.columns for col in display_columns) else project_tasks
            st.dataframe(display_df, hide_index=True)
            
            # Task timeline
            st.write("Task Timeline")
            
            # Convert dates to datetime
            project_tasks["deadline"] = pd.to_datetime(project_tasks["deadline"])
            if "completion_date" in project_tasks.columns:
                project_tasks["completion_date"] = pd.to_datetime(project_tasks["completion_date"])
            
            # Sort tasks by deadline
            sorted_tasks = project_tasks.sort_values("deadline")
            
            # Create timeline
            fig = go.Figure()
            
            for idx, task in sorted_tasks.iterrows():
                # Add task as a bar
                start_date = pd.to_datetime(task.get("date_created", task["deadline"] - pd.Timedelta(days=7)))
                end_date = task["deadline"]
                
                if task["status"] == "Completed" and "completion_date" in task and pd.notna(task["completion_date"]):
                    end_date = task["completion_date"]
                
                fig.add_trace(go.Bar(
                    x=[end_date - start_date],
                    y=[task["name"]],
                    orientation='h',
                    name=task["name"],
                    marker=dict(
                        color="green" if task["status"] == "Completed" else 
                              "yellow" if task["status"] == "In Progress" else
                              "red" if task["status"] == "On Hold" else "blue"
                    ),
                    base=start_date,
                    showlegend=False
                ))
            
            fig.update_layout(
                title=f"Task Timeline for {project_to_view}",
                xaxis=dict(
                    title="Date",
                    type="date"
                ),
                yaxis=dict(
                    title="Task",
                    categoryorder="total ascending"
                ),
                height=400 + len(sorted_tasks) * 20
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No tasks found for project '{project_to_view}'. Add tasks to this project to see details.")

# Reports page
elif page == "Reports":
    st.header("Reports & Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Weekly Reports", "Bottleneck Alerts", "Analytics"])
    
    with tab1:
        st.subheader("Weekly Reports")
        
        # Report generation
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate New Weekly Report"):
                with st.spinner("Generating weekly report..."):
                    report = generate_weekly_report()
                    st.success(f"Weekly report #{report['id']} generated!")
        
        # Report list
        if st.session_state.weekly_reports:
            report_to_view = st.selectbox("Select Report to View", 
                                         [f"{report['id']} - {report['date']} ({report['period']})" 
                                          for report in st.session_state.weekly_reports])
            
            report_id = int(report_to_view.split(" - ")[0])
            report = next((r for r in st.session_state.weekly_reports if r["id"] == report_id), None)
            
            if report:
                st.write(f"## Weekly Report #{report['id']}")
                st.write(f"**Period:** {report['period']}")
                st.write(f"**Generated on:** {report['date']}")
                
                st.write("### AI-Generated Summary")
                st.write(report["ai_summary"])
                
                # Sharing options
                st.write("### Share Report")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Send via Email", key=f"email_report_{report['id']}"):
                        success, message = send_email(
                            f"Weekly Project Summary - {report['date']}", 
                            report["ai_summary"]
                        )
                        if success:
                            st.success("Report sent via email!")
                        else:
                            st.error(message)
                
                with col2:
                    if st.button("Share on Slack", key=f"slack_report_{report['id']}"):
                        success, message = send_slack_message(
                            f"*Weekly Project Summary - {report['date']}*\n\n{report['ai_summary']}"
                        )
                        if success:
                            st.success("Report shared on Slack!")
                        else:
                            st.error(message)
        else:
            st.info("No weekly reports available. Generate a report to get started.")
    
    with tab2:
        st.subheader("Bottleneck Alerts")
        
        # Alert generation
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate New Bottleneck Alert"):
                with st.spinner("Analyzing bottlenecks..."):
                    alert = generate_bottleneck_alert()
                    if alert:
                        st.success(f"Bottleneck alert #{alert['id']} generated!")
                    else:
                        st.info("No significant bottlenecks detected.")
        
        # Alert list
        if st.session_state.bottleneck_alerts:
            alert_to_view = st.selectbox("Select Alert to View", 
                                        [f"{alert['id']} - {alert['date']}" 
                                         for alert in st.session_state.bottleneck_alerts])
            
            alert_id = int(alert_to_view.split(" - ")[0])
            alert = next((a for a in st.session_state.bottleneck_alerts if a["id"] == alert_id), None)
            
            if alert:
                st.write(f"## Bottleneck Alert #{alert['id']}")
                st.write(f"**Generated on:** {alert['date']}")
                
                st.write("### AI-Generated Analysis")
                st.write(alert["ai_summary"])
                
                # Bottleneck details
                st.write("### Detected Bottlenecks")
                for bottleneck in alert["bottlenecks"]:
                    severity_color = "red" if bottleneck["severity"] == "high" else "orange" if bottleneck["severity"] == "medium" else "blue"
                    
                    st.markdown(f"<div style='padding: 10px; background-color: {severity_color}15; border-left: 5px solid {severity_color}; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    
                    if bottleneck["type"] == "overdue_task":
                        st.write(f"**Overdue Task:** '{bottleneck['task_name']}' ({bottleneck['days_overdue']} days)")
                        st.write(f"* Assigned to: {bottleneck['assigned_to']}")
                        st.write(f"* Project: {bottleneck['project']}")
                        st.write(f"* Deadline: {bottleneck['deadline']}")
                    
                    elif bottleneck["type"] == "member_overload":
                        st.write(f"**Team Member Overload:** {bottleneck['member']}")
                        st.write(f"* Open tasks: {bottleneck['open_tasks']}")
                    
                    elif bottleneck["type"] == "project_delay":
                        st.write(f"**Project Delay:** {bottleneck['project']}")
                        st.write(f"* Completion rate: {bottleneck['completion_rate']:.1%}")
                        st.write(f"* Deadline: {bottleneck['deadline']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Sharing options
                st.write("### Share Alert")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Send via Email", key=f"email_alert_{alert['id']}"):
                        success, message = send_email(
                            f"Bottleneck Alert - {alert['date']}", 
                            alert["ai_summary"]
                        )
                        if success:
                            st.success("Alert sent via email!")
                        else:
                            st.error(message)
                
                with col2:
                    if st.button("Share on Slack", key=f"slack_alert_{alert['id']}"):
                        success, message = send_slack_message(
                            f"*Bottleneck Alert - {alert['date']}*\n\n{alert['ai_summary']}"
                        )
                        if success:
                            st.success("Alert shared on Slack!")
                        else:
                            st.error(message)
        else:
            st.info("No bottleneck alerts available. Generate an alert to get started.")
    
    with tab3:
        st.subheader("Analytics")
        
        # Time range selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.datetime.now() - datetime.timedelta(days=30))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.datetime.now())
        
        if st.session_state.tasks:
            tasks_df = pd.DataFrame(st.session_state.tasks)
            
            # Convert date columns
            date_columns = ["deadline", "date_created", "completion_date"]
            for col in date_columns:
                if col in tasks_df.columns:
                    tasks_df[col] = pd.to_datetime(tasks_df[col])
            
            # Filter by date range
            filtered_tasks = tasks_df[
                (tasks_df["date_created"] >= pd.Timestamp(start_date)) & 
                (tasks_df["date_created"] <= pd.Timestamp(end_date))
            ]
            
            if not filtered_tasks.empty:
                # Task completion trend
                st.write("#### Task Completion Trend")
                
                # Group by day and count
                if "completion_date" in filtered_tasks.columns:
                    completed_tasks = filtered_tasks[filtered_tasks["status"] == "Completed"].copy()
                    if not completed_tasks.empty:
                        completed_tasks["completion_date"] = pd.to_datetime(completed_tasks["completion_date"])
                        completion_counts = completed_tasks.groupby(completed_tasks["completion_date"].dt.date).size().reset_index()
                        completion_counts.columns = ["Date", "Completed Tasks"]
                        
                        # Create line chart
                        fig = px.line(completion_counts, x="Date", y="Completed Tasks", 
                                     title="Task Completion Trend",
                                     markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No completed tasks in the selected date range.")
                else:
                    st.info("No completion date data available.")
                
                # Task distribution by priority and status
                st.write("#### Task Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Priority distribution
                    priority_counts = filtered_tasks["priority"].value_counts().reset_index()
                    priority_counts.columns = ["Priority", "Count"]
                    
                    fig = px.pie(priority_counts, values="Count", names="Priority", 
                                title="Tasks by Priority")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Status distribution
                    status_counts = filtered_tasks["status"].value_counts().reset_index()
                    status_counts.columns = ["Status", "Count"]
                    
                    fig = px.pie(status_counts, values="Count", names="Status", 
                                title="Tasks by Status")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Time spent analysis
                st.write("#### Time Spent Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Time by project
                    project_hours = filtered_tasks.groupby("project")["hours_spent"].sum().reset_index()
                    project_hours = project_hours.sort_values("hours_spent", ascending=False)
                    
                    fig = px.bar(project_hours, x="project", y="hours_spent", 
                                title="Hours Spent by Project")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Time by team member
                    member_hours = filtered_tasks.groupby("assigned_to")["hours_spent"].sum().reset_index()
                    member_hours = member_hours.sort_values("hours_spent", ascending=False)
                    
                    fig = px.bar(member_hours, x="assigned_to", y="hours_spent", 
                                title="Hours Spent by Team Member")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Task completion time analysis
                if "completion_date" in filtered_tasks.columns and "date_created" in filtered_tasks.columns:
                    completed_tasks = filtered_tasks[filtered_tasks["status"] == "Completed"].copy()
                    if not completed_tasks.empty:
                        completed_tasks["completion_date"] = pd.to_datetime(completed_tasks["completion_date"])
                        completed_tasks["date_created"] = pd.to_datetime(completed_tasks["date_created"])
                        
                        # Calculate days to complete
                        completed_tasks["days_to_complete"] = (completed_tasks["completion_date"] - completed_tasks["date_created"]).dt.days
                        
                        # Average days by priority
                        avg_days_by_priority = completed_tasks.groupby("priority")["days_to_complete"].mean().reset_index()
                        
                        st.write("#### Task Completion Time")
                        fig = px.bar(avg_days_by_priority, x="priority", y="days_to_complete", 
                                    title="Average Days to Complete by Priority")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download analytics data
                st.download_button(
                    label="Download Analytics Data (CSV)",
                    data=filtered_tasks.to_csv(index=False),
                    file_name=f"task_analytics_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No tasks found in the selected date range.")
        else:
            st.info("No task data available. Add tasks to see analytics.")

# Settings page
elif page == "Settings":
    st.header("Settings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["API Keys", "Notification Settings", "Data Management", "About"])
    
    with tab1:
        st.subheader("API Keys")
        
        # OpenAI API key
        openai_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
        
        if st.button("Save API Key"):
            st.session_state.api_key = openai_key
            st.success("API key saved!")
        
        st.markdown("""
        > Note: The OpenAI API key is required for generating AI summaries for reports and alerts.
        > If you don't have an API key, you can get one from [OpenAI's website](https://platform.openai.com/account/api-keys).
        """)
    
    with tab2:
        st.subheader("Notification Settings")
        
        # Slack webhook
        slack_webhook = st.text_input("Slack Webhook URL", value=st.session_state.slack_webhook, type="password")
        
        if st.button("Save Slack Webhook"):
            st.session_state.slack_webhook = slack_webhook
            st.success("Slack webhook saved!")
        
        st.markdown("""
        > Note: The Slack webhook is used for sending reports and alerts to Slack.
        > Learn how to create a webhook [here](https://api.slack.com/messaging/webhooks).
        """)
        
        # Email settings
        st.subheader("Email Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smtp_server = st.text_input("SMTP Server", value=st.session_state.email_settings["smtp_server"])
            smtp_port = st.number_input("SMTP Port", value=st.session_state.email_settings["smtp_port"])
        
        with col2:
            email_username = st.text_input("Email Username", value=st.session_state.email_settings["username"])
            email_password = st.text_input("Email Password", value=st.session_state.email_settings["password"], type="password")
        
        email_recipients = st.text_area("Email Recipients (one per line)", 
                                      value="\n".join(st.session_state.email_settings["recipients"]))
        
        if st.button("Save Email Settings"):
            st.session_state.email_settings.update({
                "smtp_server": smtp_server,
                "smtp_port": smtp_port,
                "username": email_username,
                "password": email_password,
                "recipients": [email.strip() for email in email_recipients.split("\n") if email.strip()]
            })
            st.success("Email settings saved!")
        
        # Automatic notifications
        st.subheader("Automatic Notifications")
        
        auto_weekly = st.checkbox("Send weekly reports automatically", value=False)
        auto_bottleneck = st.checkbox("Send bottleneck alerts automatically", value=False)
        
        if st.button("Save Notification Settings"):
            st.success("Notification settings saved!")
            st.info("Note: In a production environment, these settings would be used to configure automated notifications.")
    
    with tab3:
        st.subheader("Data Management")
        
        # Import/Export data
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Export Data")
            
            if st.button("Export Data to JSON"):
                # Create a dictionary with all data
                export_data = {
                    "tasks": st.session_state.tasks,
                    "team_members": st.session_state.team_members,
                    "projects": st.session_state.projects,
                    "weekly_reports": st.session_state.weekly_reports,
                    "bottleneck_alerts": st.session_state.bottleneck_alerts
                }
                
                # Convert to JSON string
                json_data = json.dumps(export_data, indent=2)
                
                # Provide download button
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="ai_kpi_tracker_data.json",
                    mime="application/json"
                )
        
        with col2:
            st.write("#### Import Data")
            
            uploaded_file = st.file_uploader("Upload JSON data", type=["json"])
            
            if uploaded_file is not None:
                try:
                    import_data = json.loads(uploaded_file.getvalue())
                    
                    if st.button("Import Data"):
                        # Update session state with imported data
                        # Update session state with imported data
                        if "tasks" in import_data:
                            st.session_state.tasks = import_data["tasks"]
                        
                        if "team_members" in import_data:
                            st.session_state.team_members = import_data["team_members"]
                        
                        if "projects" in import_data:
                            st.session_state.projects = import_data["projects"]
                        
                        if "weekly_reports" in import_data:
                            st.session_state.weekly_reports = import_data["weekly_reports"]
                        
                        if "bottleneck_alerts" in import_data:
                            st.session_state.bottleneck_alerts = import_data["bottleneck_alerts"]
                        
                        # Save imported data
                        save_data()
                        
                        st.success("Data imported successfully!")
                except Exception as e:
                    st.error(f"Error importing data: {str(e)}")
        
        # Clear data option
        st.write("#### Clear Data")
        
        clear_type = st.selectbox("Select data to clear", 
                                 ["All Data", "Tasks", "Team Members", "Projects", "Reports", "Alerts"])
        
        if st.button("Clear Selected Data", type="primary"):
            if clear_type == "All Data":
                st.session_state.tasks = []
                st.session_state.team_members = []
                st.session_state.projects = []
                st.session_state.weekly_reports = []
                st.session_state.bottleneck_alerts = []
                st.success("All data has been cleared!")
            elif clear_type == "Tasks":
                st.session_state.tasks = []
                st.success("Tasks have been cleared!")
            elif clear_type == "Team Members":
                st.session_state.team_members = []
                st.success("Team members have been cleared!")
            elif clear_type == "Projects":
                st.session_state.projects = []
                st.success("Projects have been cleared!")
            elif clear_type == "Reports":
                st.session_state.weekly_reports = []
                st.success("Weekly reports have been cleared!")
            elif clear_type == "Alerts":
                st.session_state.bottleneck_alerts = []
                st.success("Bottleneck alerts have been cleared!")
            
            # Save changes
            save_data()
    
    with tab4:
        st.subheader("About AI KPI & Progress Tracker")
        
        st.markdown("""
        ### AI KPI & Progress Tracker v1.0
        
        This application helps teams track progress, manage tasks, and generate AI-powered insights.
        
        **Features:**
        - Task and project management
        - Team workload tracking
        - KPI dashboards and visualizations
        - AI-generated weekly reports
        - Bottleneck detection and alerts
        - Email and Slack notifications
        
        **Technologies used:**
        - Streamlit
        - OpenAI GPT-4o
        - Pandas
        - Plotly
        
        **Privacy Notice:**
        - All data is stored locally on your device
        - API keys are stored in session state and not shared
        
        For issues, feature requests, or contributions, please contact the developer.
        """)
        
        # Version info
        st.info("Version 1.0 - Released April 2025")
