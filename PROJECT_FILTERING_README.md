# Project-Based Filtering System

## Overview

The ML Prediction System now includes a comprehensive project-based filtering system that allows users to filter data by project ID at both pre-prediction and post-prediction phases. This enhances data exploration and analysis capabilities.

## Features

### 🔍 **Pre-Prediction Phase**
- **Automatic Project Extraction**: Extracts all unique `projectid` values from uploaded data
- **Interactive Filter Chips**: Clickable project filter buttons with visual feedback
- **Real-time Filtering**: Instantly filter preview data by selected project
- **Data Statistics**: Dynamic statistics for filtered data (row count, column stats)

### 📊 **Post-Prediction Phase**
- **Prediction Result Filtering**: Apply same filtering to prediction results
- **Preserved Columns**: Maintain all prediction columns during filtering
- **Enhanced Statistics**: Project-specific prediction statistics

### 🎯 **UI/UX Features**
- **Keyboard Navigation**: Full keyboard support for accessibility
- **Visual Active States**: Clear indication of selected projects
- **Mobile Responsive**: Optimized for all screen sizes
- **Loading States**: Visual feedback during filtering operations

## Implementation Details

### Components Structure

```
frontend/src/
├── components/
│   ├── ProjectFilters.js          # Main filter component
│   ├── DataStats.js               # Statistics display component
│   └── __tests__/
│       └── ProjectFilters.test.js # Component tests
├── App.js                         # Main app with filtering logic
└── App.css                        # Filter styling
```

### Core Components

#### 1. **ProjectFilters Component**
```jsx
<ProjectFilters
  projects={availableProjects}
  selectedProject={selectedProject}
  onSelect={handleProjectFilter}
  onReset={handleFilterReset}
  disabled={loading}
/>
```

**Features:**
- **Numerical Sorting**: Projects sorted numerically (6, 10, 12, 15, 20)
- **Keyboard Navigation**: Arrow keys, Enter, Space, Escape, Home, End
- **Accessibility**: ARIA labels, focus management, screen reader support
- **Visual Feedback**: Hover effects, active states, focus indicators

#### 2. **DataStats Component**
```jsx
<DataStats
  data={currentData}
  title={selectedProject ? `Project ${selectedProject} Statistics` : "Full Dataset Statistics"}
/>
```

**Features:**
- **Dynamic Statistics**: Row count, column count, data types
- **Missing Value Detection**: Alerts for missing data
- **Column Overview**: Unique value counts per column
- **Real-time Updates**: Statistics update with filtering

### State Management

#### Key State Variables
```javascript
// Project filtering state
const [selectedProject, setSelectedProject] = useState(null);
const [availableProjects, setAvailableProjects] = useState([]);
const [fullDataset, setFullDataset] = useState(null);
const [filteredData, setFilteredData] = useState(null);
const [filterLoading, setFilterLoading] = useState(false);
```

#### Data Flow
1. **Upload**: Extract projects from uploaded data
2. **Cache**: Store full dataset for filtering
3. **Filter**: Apply project filter to cached data
4. **Display**: Show filtered results with statistics

### Performance Optimizations

#### 1. **Memoization**
```javascript
// Memoized current data
const currentData = useMemo(() => {
  return selectedProject ? filteredData : fullDataset;
}, [selectedProject, filteredData, fullDataset]);

// Memoized display data (first/last 5 rows)
const displayData = useMemo(() => {
  if (!currentData || currentData.length === 0) return [];
  if (currentData.length <= 10) return currentData;
  
  const first5 = currentData.slice(0, 5);
  const last5 = currentData.slice(-5);
  return [...first5, ...last5];
}, [currentData]);
```

#### 2. **Non-blocking Filtering**
```javascript
const filterDataByProject = useCallback((data, projectId) => {
  setFilterLoading(true);
  
  // Use setTimeout to prevent UI blocking
  setTimeout(() => {
    const filtered = data.filter(row => 
      row.projectid && row.projectid.toString() === projectId.toString()
    );
    setFilteredData(filtered);
    setFilterLoading(false);
  }, 0);
}, []);
```

#### 3. **Callback Optimization**
```javascript
const handleProjectFilter = useCallback((projectId) => {
  setSelectedProject(projectId);
  if (projectId) {
    filterDataByProject(fullDataset, projectId);
  } else {
    setFilteredData(fullDataset);
  }
}, [fullDataset, filterDataByProject]);
```

## Usage Examples

### 1. **Upload and Filter**
```javascript
// After file upload
const projects = extractProjects(uploadedData);
setAvailableProjects(projects);
setFullDataset(uploadedData);

// User selects project
handleProjectFilter('10');
// Filters data to show only Project 10
```

### 2. **View Filtered Statistics**
```javascript
// Statistics automatically update
<DataStats 
  data={currentData} 
  title={`Project ${selectedProject} Statistics`} 
/>
// Shows: Total Rows, Total Columns, Numeric/Categorical columns
```

### 3. **Reset to Full Dataset**
```javascript
handleFilterReset();
// Shows all data with full dataset statistics
```

## Keyboard Navigation

| Key | Action |
|-----|--------|
| **Arrow Keys** | Navigate between project chips |
| **Enter/Space** | Select/deselect project |
| **Escape** | Reset to show all data |
| **Home** | Jump to first project |
| **End** | Jump to last project |

## Accessibility Features

### ARIA Support
- `role="button"` for project chips
- `aria-pressed` for selection state
- `aria-label` for screen readers
- `tabIndex` for keyboard navigation

### Focus Management
- Visual focus indicators
- Logical tab order
- Focus restoration on filter reset

### Screen Reader Support
- Descriptive labels
- State announcements
- Navigation instructions

## Mobile Responsiveness

### Breakpoint Adaptations
- **Desktop**: Full filter chips with hover effects
- **Tablet**: Compact chips with touch-friendly sizing
- **Mobile**: Stacked layout with optimized spacing

### Touch Interactions
- Larger touch targets
- Swipe-friendly chip layout
- Optimized button sizes

## CSS Architecture

### CSS Variables
```css
:root {
  --primary-blue: #0057B8;
  --box-bg: #fff;
  --text: #222;
  --box-border: #e0e6ed;
}
```

### Component Classes
- `.project-filters` - Main filter container
- `.project-chip` - Individual filter buttons
- `.data-stats` - Statistics display
- `.filter-loading` - Loading state styles

### Responsive Design
```css
@media (max-width: 768px) {
  .project-chip {
    font-size: 0.8rem;
    padding: 0.375rem 0.625rem;
  }
}
```

## Testing

### Component Tests
```javascript
// Test project selection
test('handles project selection', () => {
  fireEvent.click(screen.getByText('Project 10'));
  expect(mockOnSelect).toHaveBeenCalledWith('10');
});

// Test keyboard navigation
test('handles keyboard navigation', () => {
  fireEvent.keyDown(chip, { key: 'Enter' });
  expect(mockOnSelect).toHaveBeenCalledWith('6');
});
```

### Test Coverage
- ✅ Project chip rendering
- ✅ Selection/deselection logic
- ✅ Keyboard navigation
- ✅ Reset functionality
- ✅ Disabled states
- ✅ Numerical sorting
- ✅ Empty state handling

## Benefits

### 1. **Enhanced Data Exploration**
- Quick project-specific analysis
- Real-time data insights
- Efficient data navigation

### 2. **Improved User Experience**
- Intuitive filtering interface
- Responsive design
- Accessibility compliance

### 3. **Performance Optimization**
- Memoized computations
- Non-blocking operations
- Efficient state management

### 4. **Developer Experience**
- Modular component architecture
- Comprehensive testing
- Clear documentation

## Future Enhancements

### Potential Improvements
1. **Multi-Project Selection**: Allow selecting multiple projects
2. **Advanced Filtering**: Date ranges, value ranges, custom criteria
3. **Filter Persistence**: Remember user's last filter selection
4. **Export Filtered Data**: Export only filtered results
5. **Filter Analytics**: Track most used filters
6. **Custom Project Groups**: User-defined project groupings

### Performance Optimizations
1. **Virtual Scrolling**: For very large datasets
2. **Web Workers**: Move filtering to background threads
3. **IndexedDB**: Cache large datasets locally
4. **Lazy Loading**: Load project data on demand

## Integration Points

### Backend Integration
- Uses existing `/api/upload_uuid` endpoint
- Leverages existing prediction workflow
- No backend changes required

### Frontend Integration
- Seamless integration with existing UI
- Maintains current data flow
- Preserves all existing functionality

This project filtering system provides a powerful and user-friendly way to explore and analyze data by project, significantly enhancing the overall user experience of the ML Prediction System. 