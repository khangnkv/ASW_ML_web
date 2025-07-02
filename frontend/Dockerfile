# Use official Node.js image as base
FROM node:18

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Clear npm cache and install dependencies
RUN npm cache clean --force && \
    npm install && \
    npm ls react-scripts || echo "react-scripts not found in node_modules"

# Copy the rest of the application
COPY . .

# Install dos2unix, xsel and fix line endings in node_modules/.bin scripts
RUN apt-get update && apt-get install -y dos2unix xsel && \
    find node_modules/.bin -type f -exec dos2unix {} \; 2>/dev/null || true && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Build the application using npx directly instead of npm run script
RUN npx react-scripts build

# Install serve to run the production build  
RUN npm install -g serve@14.2.4

# Expose port
EXPOSE 3000

# Start the application with serve
CMD ["serve", "-s", "build", "-l", "3000"]
