const express = require("express");
const bodyParser = require("body-parser");
const mysql = require("mysql2/promise"); // Using promise version for async/await
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");
const nodemailer = require("nodemailer");
const bcrypt = require("bcrypt");
const http = require('http');
const httpProxy = require('http-proxy');

const app = express();
const PORT = 3000;

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "public"))); // Serve static files from 'public' folder

// Create proxy server
const proxy = httpProxy.createProxyServer({
  target: 'http://127.0.0.1:5000',
  changeOrigin: true
});

// MySQL connection setup with pool
const dbPool = mysql.createPool({
  host: "localhost",
  user: "root",
  password: "Yashwanth1234@", // Your MySQL password
  database: "user_registration",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

// Test database connection on startup
async function testConnection() {
  try {
    const connection = await dbPool.getConnection();
    console.log("Successfully connected to MySQL database");
    connection.release();
  } catch (err) {
    console.error("MySQL connection failed:", err);
  }
}

testConnection();

// Serve HTML pages
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "login.html")); // Default route to ai_monitoring.html
});

app.get("/register", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "register.html")); // Serve registration page
});

app.get("/login", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "login.html")); // Serve login page
});

app.get("/dashboard", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "dashboard.html")); // Serve dashboard page
});

app.get("/coaching_platform", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "coaching_platform.html"));
});

app.get("/home_page", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "home_page.html"));
});

app.get("/nutrition_planner", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "nutrition_planner.html"));
});

app.get("/prevention_method", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "prevention_method.html"));
});

app.get("/select_the_muscle_group", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "select_the_muscle_group.html"));
});

app.get("/startcoach", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "startcoach.html"));
});

app.get("/workout", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "workout.html"));
});

app.get("/workout_routine", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "workout_routine.html"));
});

// Authentication middleware
const authenticateUser = (req, res, next) => {
  const userId = req.headers["user-id"];
  if (!userId) {
    return res.status(401).json({ error: "Authentication required" });
  }
  next();
};

// Add startworkout route
app.get("/startworkout", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "startworkout.html"));
});

// Handle registration form submission
app.post("/register", async (req, res) => {
  const { name, age, sex, weight, height, email, password } = req.body;

  // Validate input
  if (!name || !age || !sex || !weight || !height || !email || !password) {
    return res.status(400).send("All fields are required");
  }

  const sql = `INSERT INTO users (name, age, sex, weight, height, email, password)
                 VALUES (?, ?, ?, ?, ?, ?, ?)`;

  const values = [
    name,
    parseInt(age),
    sex,
    parseFloat(weight),
    parseFloat(height),
    email,
    password,
  ];

  try {
    const [result] = await dbPool.execute(sql, values);
    console.log("User registered with ID:", result.insertId);
    res.status(200).send("Registration successful!");
  } catch (err) {
    console.error("Error inserting into database:", err);

    // Check for duplicate email error
    if (err.code === "ER_DUP_ENTRY") {
      return res.status(400).send("Email already registered");
    }

    res.status(500).send("Registration failed. Please try again.");
  }
});

// Handle login form submission (basic implementation)
app.post("/login", async (req, res) => {
  const { email, password } = req.body;

  // Validate input
  if (!email || !password) {
    return res
      .status(400)
      .json({ success: false, message: "Email and password are required" });
  }

  const sql = `SELECT * FROM users WHERE email = ? AND password = ?`;
  const values = [email, password];

  try {
    const [rows] = await dbPool.execute(sql, values);
    if (rows.length > 0) {
      // Successful login
      console.log("User logged in:", email);

      // Create user object without password
      const user = {
        id: rows[0].id,
        name: rows[0].name,
        email: rows[0].email,
        age: rows[0].age,
        sex: rows[0].sex,
        weight: rows[0].weight,
        height: rows[0].height,
      };

      res.status(200).json({ success: true, user });
    } else {
      res
        .status(401)
        .json({ success: false, message: "Invalid email or password" });
    }
  } catch (err) {
    console.error("Error during login:", err);
    res
      .status(500)
      .json({ success: false, message: "Login failed. Please try again." });
  }
});

// Workout monitoring endpoints
app.post("/api/workout/start", authenticateUser, async (req, res) => {
  try {
    const { userId, workoutType } = req.body;

    if (!userId || !workoutType) {
      return res.status(400).json({ error: "Missing required parameters" });
    }

    // Generate a unique session ID
    const sessionId = Date.now().toString();

    // Start the Python script for workout monitoring
    const pythonProcess = spawn("python", [
      "Python/workout_monitering.py",
      userId,
      workoutType,
    ]);

    // Store the process and session information
    workoutProcesses.set(sessionId, pythonProcess);
    workoutSessions.set(sessionId, {
      userId,
      workoutType,
      startTime: Date.now(),
      data: {
        repCount: 0,
        duration: 0,
        calories: 0,
      },
    });

    // Handle Python script output
    pythonProcess.stdout.on("data", (data) => {
      try {
        const workoutData = JSON.parse(data.toString());
        const session = workoutSessions.get(sessionId);
        if (session) {
          session.data = workoutData;
        }
      } catch (error) {
        console.error("Error parsing workout data:", error);
      }
    });

    // Handle Python script errors
    pythonProcess.stderr.on("data", (data) => {
      console.error(`Python script error: ${data}`);
    });

    // Handle Python script exit
    pythonProcess.on("close", (code) => {
      console.log(`Python script exited with code ${code}`);
      workoutProcesses.delete(sessionId);
      workoutSessions.delete(sessionId);
    });

    res.json({ sessionId });
  } catch (error) {
    console.error("Error starting workout:", error);
    res.status(500).json({ error: "Failed to start workout" });
  }
});

app.get("/api/workout/data", (req, res) => {
  const sessionId = req.query.sessionId;
  const session = workoutSessions.get(sessionId);

  if (!session) {
    return res.status(404).json({ error: "Workout session not found" });
  }

  res.json(session.data);
});

app.post("/api/workout/end", async (req, res) => {
  try {
    const { sessionId } = req.body;
    const session = workoutSessions.get(sessionId);

    if (!session) {
      return res.status(404).json({ error: "Workout session not found" });
    }

    // Kill the Python process
    const pythonProcess = workoutProcesses.get(sessionId);
    if (pythonProcess) {
      pythonProcess.kill();
      workoutProcesses.delete(sessionId);
    }

    // Save workout data to database
    const { userId, workoutType, data } = session;
    const [result] = await dbPool.execute(
      "INSERT INTO `workout_sessions` (user_id, exercise_type, reps, duration) VALUES (?, ?, ?, ?)",
      [userId, workoutType, data.repCount || 0, data.duration || 0]
    );

    // Clean up session data
    workoutSessions.delete(sessionId);

    res.json({
      success: true,
      message: "Workout saved successfully",
      workoutId: result.insertId,
    });
  } catch (error) {
    console.error("Error ending workout:", error);
    res.status(500).json({ error: "Failed to save workout data" });
  }
});

// Get workout sessions for a user
app.get("/api/workout/sessions", async (req, res) => {
  const { userId } = req.query;
  try {
    const sql = `SELECT * FROM workout_sessions WHERE user_id = ? ORDER BY timestamp DESC`;
    const [rows] = await dbPool.execute(sql, [userId]);
    res.status(200).json(rows);
  } catch (err) {
    console.error("Error fetching workout sessions:", err);
    res.status(500).json({ error: "Failed to fetch workout sessions" });
  }
});

// Get workout templates for a user
app.get("/api/workout/templates", async (req, res) => {
  const { userId } = req.query;
  try {
    const sql = `SELECT * FROM workout_templates WHERE user_id = ? OR is_public = 1 ORDER BY created_at DESC`;
    const [rows] = await dbPool.execute(sql, [userId]);
    res.status(200).json(rows);
  } catch (err) {
    console.error("Error fetching workout templates:", err);
    res.status(500).json({ error: "Failed to fetch workout templates" });
  }
});

// Food analysis endpoint
app.post("/api/nutrition/analyze", async (req, res) => {
  try {
    const { imageData } = req.body;
    if (!imageData) {
      return res.status(400).json({ error: "No image data provided" });
    }

    // Convert base64 to buffer
    const base64Data = imageData.replace(/^data:image\/\w+;base64,/, "");
    const buffer = Buffer.from(base64Data, "base64");

    // Create form data for Flask server
    const formData = new FormData();
    formData.append(
      "image",
      new Blob([buffer], { type: "image/jpeg" }),
      "food.jpg"
    );

    // Send request to Flask server
    const response = await fetch("http://localhost:5000/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Failed to analyze image");
    }

    const result = await response.json();
    res.json(result);
  } catch (error) {
    console.error("Error analyzing food:", error);
    res.status(500).json({ error: "Failed to analyze food image" });
  }
});

// Get nutrition logs for a user
app.get("/api/nutrition/logs", async (req, res) => {
  const { userId } = req.query;
  try {
    const sql = `SELECT * FROM nutrition_logs WHERE user_id = ? ORDER BY timestamp DESC`;
    const [rows] = await dbPool.execute(sql, [userId]);
    res.status(200).json(rows);
  } catch (err) {
    console.error("Error fetching nutrition logs:", err);
    res.status(500).json({ error: "Failed to fetch nutrition logs" });
  }
});

// Add a new nutrition log
app.post("/api/nutrition/logs", async (req, res) => {
  const { userId, meal_type, food_items, calories, protein, carbs, fat } =
    req.body;
  try {
    const sql = `INSERT INTO nutrition_logs (user_id, meal_type, food_items, calories, protein, carbs, fat, timestamp) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, NOW())`;
    const [result] = await dbPool.execute(sql, [
      userId,
      meal_type,
      JSON.stringify(food_items),
      calories,
      protein,
      carbs,
      fat,
    ]);

    // Get the newly created log
    const [newLog] = await dbPool.execute(
      "SELECT * FROM nutrition_logs WHERE id = ?",
      [result.insertId]
    );

    res.status(200).json(newLog[0]);
  } catch (err) {
    console.error("Error adding nutrition log:", err);
    res.status(500).json({ error: "Failed to add nutrition log" });
  }
});

// Coaching Platform Endpoints
app.get("/api/coaches", async (req, res) => {
  try {
    const [rows] = await dbPool.execute("SELECT * FROM coaches");
    res.status(200).json(rows);
  } catch (err) {
    console.error("Error fetching coaches:", err);
    res.status(500).json({ error: "Failed to fetch coaches" });
  }
});

app.post("/api/coaching/session", async (req, res) => {
  const { userId, coachId, sessionType } = req.body;
  try {
    const sql = `INSERT INTO coaching_sessions (user_id, coach_id, session_type, timestamp) 
                     VALUES (?, ?, ?, NOW())`;
    await dbPool.execute(sql, [userId, coachId, sessionType]);
    res.status(200).json({ message: "Coaching session scheduled" });
  } catch (err) {
    console.error("Error scheduling coaching session:", err);
    res.status(500).json({ error: "Failed to schedule coaching session" });
  }
});

// User Profile Endpoints
app.get("/api/profile/:userId", async (req, res) => {
  try {
    const [rows] = await dbPool.execute("SELECT * FROM users WHERE id = ?", [
      req.params.userId,
    ]);
    if (rows.length > 0) {
      res.status(200).json(rows[0]);
    } else {
      res.status(404).json({ error: "User not found" });
    }
  } catch (err) {
    console.error("Error fetching profile:", err);
    res.status(500).json({ error: "Failed to fetch profile" });
  }
});

app.put("/api/profile/:userId", async (req, res) => {
  const { name, age, weight, height } = req.body;
  try {
    const sql = `UPDATE users SET name = ?, age = ?, weight = ?, height = ? WHERE id = ?`;
    await dbPool.execute(sql, [name, age, weight, height, req.params.userId]);
    res.status(200).json({ message: "Profile updated successfully" });
  } catch (err) {
    console.error("Error updating profile:", err);
    res.status(500).json({ error: "Failed to update profile" });
  }
});

// Add nodemailer for sending emails
const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: "sahilgowda204@gmail.com", // Your email
    pass: "yqkyimwuhgsxwaqn", // Your app password
  },
});

// Store OTPs temporarily (in production, use Redis or similar)
const otpStore = new Map();

// Generate OTP
function generateOTP() {
  return Math.floor(100000 + Math.random() * 900000).toString();
}

// Send OTP email
async function sendOTPEmail(email, otp) {
  const mailOptions = {
    from: "sahilgowda204@gmail.com",
    to: email,
    subject: "Password Reset OTP - Smart Flex",
    html: `
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                <h1 style="color: #185a9d; text-align: center;">Password Reset OTP</h1>
                <p>Hello,</p>
                <p>You have requested to reset your password for your Smart Flex account.</p>
                <p>Your OTP for password reset is: <strong style="font-size: 24px; color: #43cea2; letter-spacing: 2px;">${otp}</strong></p>
                <p>This OTP will expire in 5 minutes.</p>
                <p>If you didn't request this, please ignore this email.</p>
                <p style="margin-top: 30px; text-align: center; color: #666; font-size: 12px;">© 2023 Smart Flex. All rights reserved.</p>
            </div>
        `,
  };

  try {
    await transporter.sendMail(mailOptions);
    return true;
  } catch (error) {
    console.error("Error sending email:", error);
    return false;
  }
}

// Forgot Password - Send OTP
app.post("/api/auth/forgot-password", async (req, res) => {
  const { email } = req.body;

  if (!email) {
    return res.status(400).json({ message: "Email is required" });
  }

  try {
    // Check if user exists
    const [users] = await dbPool.execute(
      "SELECT * FROM users WHERE email = ?",
      [email]
    );

    if (users.length === 0) {
      return res.status(404).json({ message: "User not found" });
    }

    // Generate and store OTP
    const otp = generateOTP();
    otpStore.set(email, {
      otp,
      timestamp: Date.now(),
      attempts: 0,
    });

    // Send OTP email
    const emailSent = await sendOTPEmail(email, otp);

    if (!emailSent) {
      return res.status(500).json({ message: "Failed to send OTP email" });
    }

    res.json({ message: "OTP sent successfully" });
  } catch (error) {
    console.error("Error in forgot password:", error);
    res.status(500).json({ message: "Internal server error" });
  }
});

// Verify OTP
app.post("/api/auth/verify-otp", (req, res) => {
  const { email, otp } = req.body;

  if (!email || !otp) {
    return res.status(400).json({ message: "Email and OTP are required" });
  }

  const storedData = otpStore.get(email);

  if (!storedData) {
    return res.status(400).json({ message: "OTP expired or not found" });
  }

  // Check if OTP is expired (5 minutes)
  if (Date.now() - storedData.timestamp > 5 * 60 * 1000) {
    otpStore.delete(email);
    return res.status(400).json({ message: "OTP expired" });
  }

  // Check if too many attempts
  if (storedData.attempts >= 3) {
    otpStore.delete(email);
    return res
      .status(400)
      .json({ message: "Too many attempts. Please request a new OTP" });
  }

  // Verify OTP
  if (storedData.otp !== otp) {
    storedData.attempts++;
    return res.status(400).json({ message: "Invalid OTP" });
  }

  res.json({ message: "OTP verified successfully" });
});

// Reset Password
app.post("/api/auth/reset-password", async (req, res) => {
  const { email, otp, newPassword } = req.body;

  if (!email || !otp || !newPassword) {
    return res.status(400).json({ message: "All fields are required" });
  }

  const storedData = otpStore.get(email);

  if (!storedData || storedData.otp !== otp) {
    return res.status(400).json({ message: "Invalid OTP" });
  }

  try {
    // Hash the new password
    const hashedPassword = await bcrypt.hash(newPassword, 10);

    // Update password in database
    const sql = `UPDATE users SET password = ? WHERE email = ?`;
    await dbPool.execute(sql, [hashedPassword, email]);

    // Clear OTP
    otpStore.delete(email);

    res.json({ message: "Password reset successfully" });
  } catch (error) {
    console.error("Error resetting password:", error);
    res.status(500).json({ message: "Internal server error" });
  }
});

// Chatbot endpoint
app.post("/api/chat", async (req, res) => {
  const { userId, message, workoutSession } = req.body;

  if (!userId || !message) {
    return res.status(400).json({ error: "Missing required parameters" });
  }

  try {
    let response;

    // If there's an active workout session, include workout data in the response
    if (workoutSession) {
      const workoutData = workoutSessions.get(workoutSession);
      if (workoutData) {
        response = generateWorkoutResponse(message, workoutData);
      } else {
        response =
          "I'm having trouble accessing your workout data. Please try again.";
      }
    } else {
      response = generateGeneralResponse(message);
    }

    res.json({ response });
  } catch (error) {
    console.error("Error processing chat message:", error);
    res.status(500).json({ error: "Failed to process chat message" });
  }
});

// Helper functions for chatbot responses
function generateWorkoutResponse(message, workoutData) {
  const lowerMessage = message.toLowerCase();

  if (lowerMessage.includes("form") || lowerMessage.includes("posture")) {
    let formMessage = "Great job maintaining proper form!";
    if (workoutData.formScore < 80) {
      formMessage =
        workoutData.formScore >= 60
          ? "Try to focus on your posture and alignment."
          : "Please adjust your form to prevent injury.";
    }
    return `Your current form score is ${workoutData.formScore}%. ${formMessage}`;
  } else if (
    lowerMessage.includes("calories") ||
    lowerMessage.includes("burn")
  ) {
    return `You've burned approximately ${workoutData.calories} calories so far. Keep up the good work!`;
  } else if (
    lowerMessage.includes("time") ||
    lowerMessage.includes("duration")
  ) {
    let timeMessage = "Just getting started! Keep going!";
    if (workoutData.duration >= "00:15:00") {
      timeMessage = "You're doing amazing! Push yourself a bit more!";
    } else if (workoutData.duration >= "00:05:00") {
      timeMessage = "You're making great progress!";
    }
    return `You've been working out for ${workoutData.duration}. ${timeMessage}`;
  } else {
    return generateGeneralResponse(message);
  }
}

function generateGeneralResponse(message) {
  const lowerMessage = message.toLowerCase();

  if (lowerMessage.includes("hello") || lowerMessage.includes("hi")) {
    return "Hello! I'm your AI fitness coach. How can I help you today?";
  } else if (lowerMessage.includes("help")) {
    return "I can help you with workout form, track your progress, and provide motivation. What would you like to know?";
  } else if (
    lowerMessage.includes("motivation") ||
    lowerMessage.includes("encourage")
  ) {
    return "You're doing great! Remember, every rep counts towards your fitness goals. Keep pushing yourself!";
  } else {
    return "I'm here to support your fitness journey. Would you like to start a workout session or ask about specific exercises?";
  }
}

// Store active workout processes and sessions
const workoutProcesses = new Map();
const workoutSessions = new Map();

// Session verification endpoint
app.post("/api/auth/verify-session", async (req, res) => {
  const { userId, email } = req.body;

  if (!userId || !email) {
    return res.status(400).json({ error: "Missing required fields" });
  }

  try {
    // Check if user exists in database
    const [users] = await dbPool.execute(
      "SELECT id, email FROM users WHERE id = ? AND email = ?",
      [userId, email]
    );

    if (users.length === 0) {
      return res.status(401).json({ error: "Invalid session" });
    }

    // Session is valid
    return res.status(200).json({
      success: true,
      user: {
        id: users[0].id,
        email: users[0].email,
      },
    });
  } catch (error) {
    console.error("Error verifying session:", error);
    return res.status(500).json({ error: "Server error" });
  }
});

// Custom Workouts Endpoints
app.post("/api/workout/save-custom", async (req, res) => {
  const userId = req.headers["x-user-id"];
  if (!userId) {
    return res
      .status(401)
      .json({ success: false, message: "User not authenticated" });
  }

  const { name, description, exercises } = req.body;
  if (
    !name ||
    !exercises ||
    !Array.isArray(exercises) ||
    exercises.length === 0
  ) {
    return res
      .status(400)
      .json({ success: false, message: "Invalid workout data" });
  }

  try {
    const connection = await dbPool.getConnection();
    try {
      await connection.beginTransaction();

      // Insert the workout
      const [result] = await connection.execute(
        "INSERT INTO custom_workouts (user_id, name, description, exercises) VALUES (?, ?, ?, ?)",
        [userId, name, description, JSON.stringify(exercises)]
      );

      await connection.commit();
      res.json({ success: true, workoutId: result.insertId });
    } catch (error) {
      await connection.rollback();
      throw error;
    } finally {
      connection.release();
    }
  } catch (error) {
    console.error("Error saving custom workout:", error);
    res.status(500).json({ success: false, message: "Failed to save workout" });
  }
});

app.get("/api/workout/custom/:userId", async (req, res) => {
  const userId = req.params.userId;
  if (!userId) {
    return res
      .status(400)
      .json({ success: false, message: "User ID is required" });
  }

  try {
    const [workouts] = await dbPool.execute(
      "SELECT * FROM custom_workouts WHERE user_id = ? ORDER BY created_at DESC",
      [userId]
    );

    // Parse the exercises JSON for each workout
    const formattedWorkouts = workouts.map((workout) => ({
      ...workout,
      exercises: JSON.parse(workout.exercises),
    }));

    res.json(formattedWorkouts);
  } catch (error) {
    console.error("Error retrieving custom workouts:", error);
    res
      .status(500)
      .json({ success: false, message: "Failed to retrieve workouts" });
  }
});

// Add bone density routes
app.get("/bone_density.html", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "bone_density.html"));
});

// Proxy bone density API requests to Flask server
app.all("/api/bone_density/*", (req, res) => {
  proxy.web(req, res, {}, (err) => {
    if (err.code === 'ECONNREFUSED') {
      res.status(503).json({ error: 'Bone density service is not available' });
    } else {
      res.status(500).json({ error: 'Failed to connect to bone density service' });
    }
  });
});

// Error handling for proxy
proxy.on('error', (err, req, res) => {
  console.error('Proxy error:', {
    code: err.code,
    message: err.message,
    stack: err.stack
  });
  
  if (err.code === 'ECONNREFUSED') {
    console.error('Flask server is not running or not accessible');
    res.status(503).json({ error: 'Bone density service is not available. Please try again later.' });
  } else {
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send("Something went wrong!");
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
