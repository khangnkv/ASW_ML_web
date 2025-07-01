# Use official Node.js image as base
FROM node:20-slim

# Set working directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package.json ./
COPY package-lock.json ./
RUN npm install --legacy-peer-deps

# Copy frontend source
COPY . .

# Build frontend
RUN npm run build

# Expose frontend port
EXPOSE 3000

# Start frontend
CMD ["npm", "start"]
