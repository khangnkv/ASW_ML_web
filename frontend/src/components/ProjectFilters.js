import React, { useState, useCallback, useMemo } from 'react';
import { Badge, Button, Row, Col } from 'react-bootstrap';
import { FiFilter, FiX } from 'react-icons/fi';

const ProjectFilters = ({ 
  projects = [], 
  selectedProject, 
  onSelect, 
  onReset, 
  disabled = false,
  className = '' 
}) => {
  const [focusedIndex, setFocusedIndex] = useState(-1);

  // Memoize sorted projects for better performance
  const sortedProjects = useMemo(() => {
    return [...projects].sort((a, b) => {
      // Sort numerically if possible, otherwise alphabetically
      const aNum = parseInt(a);
      const bNum = parseInt(b);
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return aNum - bNum;
      }
      return a.toString().localeCompare(b.toString());
    });
  }, [projects]);

  const handleProjectClick = useCallback((projectId) => {
    if (disabled) return;
    
    if (selectedProject === projectId) {
      onReset();
    } else {
      onSelect(projectId);
    }
  }, [selectedProject, onSelect, onReset, disabled]);

  const handleKeyDown = useCallback((e, projectId, index) => {
    if (disabled) return;

    switch (e.key) {
      case 'Enter':
      case ' ':
        e.preventDefault();
        handleProjectClick(projectId);
        break;
      case 'ArrowRight':
        e.preventDefault();
        setFocusedIndex(Math.min(index + 1, sortedProjects.length - 1));
        break;
      case 'ArrowLeft':
        e.preventDefault();
        setFocusedIndex(Math.max(index - 1, 0));
        break;
      case 'Home':
        e.preventDefault();
        setFocusedIndex(0);
        break;
      case 'End':
        e.preventDefault();
        setFocusedIndex(sortedProjects.length - 1);
        break;
      case 'Escape':
        e.preventDefault();
        onReset();
        setFocusedIndex(-1);
        break;
    }
  }, [handleProjectClick, sortedProjects.length, onReset, disabled]);

  const handleResetClick = useCallback(() => {
    if (disabled) return;
    onReset();
    setFocusedIndex(-1);
  }, [onReset, disabled]);

  if (sortedProjects.length === 0) {
    return null;
  }

  return (
    <div className={`project-filters ${className}`}>
      <div className="filter-header mb-3">
        <Row className="align-items-center">
          <Col>
            <h6 className="mb-0">
              <FiFilter className="me-2" />
              Filter by Project
              {selectedProject && (
                <Badge bg="primary" className="ms-2">
                  {selectedProject}
                </Badge>
              )}
            </h6>
            <small className="text-muted">
              {selectedProject 
                ? `Showing data for Project ${selectedProject}`
                : `Showing all ${sortedProjects.length} projects`
              }
            </small>
          </Col>
          <Col xs="auto">
            <Button
              variant="outline-secondary"
              size="sm"
              onClick={handleResetClick}
              disabled={disabled || !selectedProject}
              className="reset-btn"
            >
              <FiX className="me-1" />
              Show All
            </Button>
          </Col>
        </Row>
      </div>

      <div className="filter-chips">
        {sortedProjects.map((projectId, index) => {
          const isSelected = selectedProject === projectId;
          const isFocused = focusedIndex === index;
          
          return (
            <Badge
              key={projectId}
              bg={isSelected ? 'primary' : 'light'}
              text={isSelected ? 'white' : 'dark'}
              className={`project-chip ${isSelected ? 'selected' : ''} ${isFocused ? 'focused' : ''}`}
              onClick={() => handleProjectClick(projectId)}
              onKeyDown={(e) => handleKeyDown(e, projectId, index)}
              tabIndex={0}
              role="button"
              aria-pressed={isSelected}
              aria-label={`Filter by Project ${projectId}`}
              disabled={disabled}
            >
              Project {projectId}
              {isSelected && <FiX className="ms-1" />}
            </Badge>
          );
        })}
      </div>

      <div className="filter-info mt-2">
        <small className="text-muted">
          Use arrow keys to navigate, Enter to select, Escape to reset
        </small>
      </div>
    </div>
  );
};

export default ProjectFilters; 