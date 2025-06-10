# 📊 AI KPI & Progress Tracker

A Streamlit-based application for managing team tasks, projects, and KPIs, with AI-powered reporting, bottleneck detection, and automated notifications via email and Slack.

---

## 🚀 Features

- **Task & Project Management:** Add, edit, filter, and track tasks and projects.
- **Team Management:** Manage team members, roles, and workloads.
- **KPI Dashboards:** Visualize key performance indicators, completion rates, and project status.
- **AI-Generated Reports:** Use OpenAI GPT-4o to generate weekly summaries and bottleneck alerts.
- **Bottleneck Detection:** Identify overdue tasks, overloaded team members, and project delays.
- **Analytics:** Analyze trends, time spent, and completion times.
- **Notifications:** Send reports and alerts via email and Slack.
- **Data Import/Export:** Backup and restore all data in JSON format.
- **Settings:** Manage API keys, notification settings, and data.

---

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/)
- [OpenAI GPT-4o](https://platform.openai.com/)
- [Pandas](https://pandas.pydata.org/)
- [Plotly](https://plotly.com/python/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Requests](https://docs.python-requests.org/)
- [smtplib/email](https://docs.python.org/3/library/smtplib.html) (for email notifications)

---

## 📦 Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/shahmohsin1230/Progress_tracker.git
   cd ai-kpi-progress-tracker
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up API keys:**
   - Create a `.env` file in the root directory:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```
   - (Optional) Set up your email SMTP and Slack webhook in the app's Settings tab.

---

## ⚡ Usage

1. **Run the app:**
   ```sh
   streamlit run a.py
   ```

2. **Navigate the app:**
   - Use the sidebar to access Dashboard, Tasks, Team, Projects, Reports, and Settings.
   - Configure your OpenAI API key in the Settings tab to enable AI summaries.
   - Add your team, projects, and tasks.
   - Generate and distribute reports as needed.

---

## 🔒 Privacy Notice

- All data is stored locally on your device as JSON files.
- API keys are stored in Streamlit session state and are not shared.

---

## 📝 Data Files

- `tasks.json` — Task data
- `team_members.json` — Team member data
- `projects.json` — Project data
- `weekly_reports.json` — Weekly AI-generated reports
- `bottleneck_alerts.json` — Bottleneck alert reports

---

## 🛡️ Security

- Email and Slack credentials are not stored in code.
- All sensitive settings are managed via the Settings tab.

---

## 🧑‍💻 Contributing

Pull requests and feature suggestions are welcome! Please open an issue or submit a PR.

---

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

