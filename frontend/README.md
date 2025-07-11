# Frontend - DeFi Q&A Bot

A modern React TypeScript frontend with real-time WebSocket streaming and responsive design for the DeFi Q&A Bot.

## 🚀 Features

### Core Features
- **⚡ Real-time Streaming**: Word-by-word response streaming via WebSocket
- **🔄 Auto-Reconnection**: Automatic WebSocket reconnection with exponential backoff
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **🎯 TypeScript**: Full type safety and modern development experience
- **🔍 Connection Status**: Real-time connection status monitoring
- **🎨 Modern UI**: Clean, intuitive interface with loading states and animations

### Technical Features
- **WebSocket Integration**: Bidirectional real-time communication
- **Error Handling**: Comprehensive error states with user-friendly messages
- **Performance Optimized**: Efficient rendering and state management
- **Accessibility**: WCAG compliant with proper ARIA labels
- **Progressive Enhancement**: Works with and without JavaScript

## 📋 Quick Start

### Prerequisites
- **Node.js 18+**
- **npm 9+** or **yarn 1.22+**

### Installation
```bash
# Clone and navigate
git clone <repository-url>
cd Chat_bot/frontend

# Install dependencies
npm install
# or
yarn install
```

### Development
```bash
# Start development server
npm start
# or
yarn start

# Application will open at http://localhost:3000
```

### Build
```bash
# Create production build
npm run build
# or
yarn build

# Build output will be in ./build/
```

### Test
```bash
# Run tests
npm test
# or
yarn test

# Run tests with coverage
npm run test:coverage
# or
yarn test:coverage
```

## 🏗️ Architecture

### Component Structure
```
src/
├── App.tsx                 # Main application component
├── App.css                 # Application styles
├── index.tsx              # Application entry point
├── index.css              # Global styles
├── components/            # Reusable components (future)
├── hooks/                 # Custom React hooks (future)
├── services/              # API and WebSocket services (future)
├── types/                 # TypeScript type definitions (future)
└── utils/                 # Utility functions (future)
```

### Current Implementation
The current implementation is a single-component application in `App.tsx` that includes:

- **WebSocket Connection Management**: Automatic connection, reconnection, and error handling
- **Real-time Message Handling**: Word-by-word streaming display
- **User Interface**: Question input, response display, and status indicators
- **State Management**: React hooks for local state management

### WebSocket Integration
```typescript
// WebSocket connection with auto-reconnection
const connectWebSocket = () => {
  const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${backendHost}/ws/ask`;
  const ws = new WebSocket(wsUrl);
  
  ws.onopen = () => {
    setConnectionStatus('Connected');
    setReconnectAttempts(0);
  };
  
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleWebSocketMessage(message);
  };
  
  ws.onclose = () => {
    setConnectionStatus('Disconnected');
    attemptReconnection();
  };
};
```

## 🛠️ Development

### Available Scripts

#### `npm start`
Runs the app in development mode at [http://localhost:3000](http://localhost:3000)
- Hot reloading enabled
- Error overlay for development
- Proxy configured for backend API

#### `npm test`
Launches the test runner in interactive watch mode
- Runs Jest tests
- Supports snapshot testing
- Coverage reports available

#### `npm run build`
Builds the app for production to the `build` folder
- Optimized for best performance
- Code splitting and tree shaking
- Minified and hashed filenames

#### `npm run eject`
**⚠️ One-way operation!** Ejects from Create React App for full control over configuration.

### Environment Configuration

#### Development Environment
The frontend automatically detects the backend URL:
- **Development**: `localhost:8000` (default)
- **Production**: Same domain as frontend

#### Environment Variables
Create `.env.local` for local overrides:
```env
# Backend URL override (optional)
REACT_APP_BACKEND_URL=http://localhost:8000

# Enable/disable features (future)
REACT_APP_ENABLE_ANALYTICS=false
REACT_APP_DEBUG_MODE=true
```

### Backend Integration

#### API Endpoints Used
- **WebSocket**: `ws://localhost:8000/ws/ask` - Real-time question/answer streaming
- **HTTP Fallback**: `POST /ask` - Direct question endpoint (if WebSocket fails)
- **Health Check**: `GET /health` - Backend status verification

#### Message Types
The frontend handles these WebSocket message types:
```typescript
// Word streaming
{ type: 'word', content: 'token', confidence: 0.85 }

// Metadata
{ type: 'metadata', session_id: 'abc123', total_words: 45 }

// Completion
{ type: 'complete', final_confidence: 0.85 }

// Errors
{ type: 'error', message: 'Error description' }
```

### State Management

#### Current State Structure
```typescript
// Connection state
const [connectionStatus, setConnectionStatus] = useState('Connecting');
const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);

// UI state  
const [question, setQuestion] = useState('');
const [response, setResponse] = useState('');
const [isLoading, setIsLoading] = useState(false);

// Streaming state
const [confidence, setConfidence] = useState<number | null>(null);
const [sessionId, setSessionId] = useState<string | null>(null);
```

#### Future Enhancements
Consider migrating to:
- **Context API**: For complex state sharing
- **Redux Toolkit**: For predictable state management
- **Zustand**: For lightweight state management

### Styling

#### Current Approach
- **CSS Modules**: Component-scoped styles in `App.css`
- **Responsive Design**: Mobile-first approach with media queries
- **CSS Variables**: For consistent theming

#### CSS Structure
```css
/* Global styles */
:root {
  --primary-color: #007bff;
  --success-color: #28a745;
  --error-color: #dc3545;
  --background-color: #f8f9fa;
}

/* Component styles */
.App { /* Main container */ }
.question-form { /* Form styling */ }
.response-container { /* Response display */ }
.status-indicator { /* Connection status */ }
```

#### Future Styling Options
- **Styled Components**: CSS-in-JS with TypeScript support
- **Tailwind CSS**: Utility-first CSS framework
- **Material-UI**: Component library with consistent design
- **Chakra UI**: Modular and accessible component library

## 🎨 UI/UX Features

### Current Features
- **Real-time Typing Effect**: Words appear as they're streamed from the backend
- **Connection Status Indicator**: Visual feedback for WebSocket connection state
- **Loading States**: Clear indication when processing questions
- **Error Handling**: User-friendly error messages with retry options
- **Responsive Layout**: Works on all screen sizes

### Accessibility
- **Keyboard Navigation**: Full keyboard support
- **Screen Reader Support**: Proper ARIA labels and semantic HTML
- **High Contrast**: Readable colors and proper contrast ratios
- **Focus Management**: Clear focus indicators

### Performance
- **Optimized Rendering**: Efficient state updates for streaming text
- **Memory Management**: Proper WebSocket cleanup
- **Code Splitting**: Bundle optimization (via Create React App)

## 🧪 Testing

### Test Structure
```bash
src/
├── App.test.tsx           # Main component tests
├── setupTests.ts          # Test configuration
└── __tests__/            # Additional test files (future)
```

### Testing Strategy

#### Unit Tests
- Component rendering
- User interaction handling
- State management
- Utility functions

#### Integration Tests
- WebSocket communication
- API integration
- Error handling flows
- User workflow testing

#### Example Test
```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';

test('submits question and displays response', async () => {
  render(<App />);
  
  const input = screen.getByPlaceholderText(/ask a question/i);
  const button = screen.getByRole('button', { name: /ask/i });
  
  fireEvent.change(input, { target: { value: 'What is DeFi?' } });
  fireEvent.click(button);
  
  // Assert loading state
  expect(screen.getByText(/processing/i)).toBeInTheDocument();
});
```

### Running Tests
```bash
# Interactive test runner
npm test

# Run all tests once
npm test -- --watchAll=false

# Coverage report
npm test -- --coverage --watchAll=false

# Specific test file
npm test App.test.tsx
```

## 🚀 Deployment

### Build Process
```bash
# Create production build
npm run build

# Serve locally to test
npx serve -s build
```

### Deployment Options

#### Static Hosting
- **Vercel**: `vercel deploy`
- **Netlify**: Drag and drop `build` folder
- **GitHub Pages**: Configure in repository settings
- **AWS S3**: Upload build files to S3 bucket

#### Docker Deployment
```dockerfile
# Build stage
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . ./
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Environment-Specific Builds
```bash
# Development build
REACT_APP_ENV=development npm run build

# Staging build  
REACT_APP_ENV=staging npm run build

# Production build
REACT_APP_ENV=production npm run build
```

### Production Considerations
- **HTTPS**: Ensure WebSocket connects to `wss://` in production
- **CORS**: Backend must allow frontend domain
- **Error Boundaries**: Implement React error boundaries
- **Performance Monitoring**: Consider tools like Sentry or LogRocket

## 📦 Dependencies

### Core Dependencies
```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "typescript": "^4.9.5",
  "web-vitals": "^2.1.4"
}
```

### Development Dependencies
```json
{
  "@testing-library/jest-dom": "^5.16.4",
  "@testing-library/react": "^13.4.0",
  "@testing-library/user-event": "^13.5.0",
  "@types/jest": "^27.5.2",
  "@types/node": "^16.18.14",
  "@types/react": "^18.0.28",
  "@types/react-dom": "^18.0.11"
}
```

### Future Dependencies
Consider adding:
- **React Router**: For multi-page navigation
- **Axios**: For HTTP client with interceptors  
- **React Query**: For server state management
- **Formik/React Hook Form**: For form handling
- **React Testing Library**: Enhanced testing utilities

## 🔧 Configuration

### TypeScript Configuration
```json
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
```

### Proxy Configuration
Development proxy is configured in `package.json`:
```json
{
  "proxy": "http://localhost:8000"
}
```

## 🤝 Contributing

### Development Workflow
1. Create feature branch
2. Make changes following code standards
3. Add/update tests
4. Ensure all tests pass
5. Submit pull request

### Code Standards
- **TypeScript**: Full type safety required
- **ESLint**: Follow configured linting rules
- **Prettier**: Automatic code formatting
- **Component Structure**: Functional components with hooks
- **Testing**: Test all new features

### Component Guidelines
```typescript
// Preferred component structure
import React, { useState, useEffect } from 'react';
import './ComponentName.css';

interface ComponentProps {
  title: string;
  onAction: () => void;
}

const ComponentName: React.FC<ComponentProps> = ({ title, onAction }) => {
  const [state, setState] = useState(initialValue);
  
  useEffect(() => {
    // Effect logic
  }, [dependencies]);
  
  return (
    <div className="component-name">
      {/* Component JSX */}
    </div>
  );
};

export default ComponentName;
```

## 📚 Learn More

### React Resources
- [React Documentation](https://reactjs.org/)
- [TypeScript with React](https://react-typescript-cheatsheet.netlify.app/)
- [Create React App Documentation](https://facebook.github.io/create-react-app/docs/getting-started)

### WebSocket Resources
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Real-time Web Apps](https://web.dev/websockets/)

### Testing Resources
- [Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [Jest Documentation](https://jestjs.io/docs/getting-started)

---

**For general project information, see the main [README.md](../README.md)**
**For backend information, see [backend/README.md](../backend/README.md)**
