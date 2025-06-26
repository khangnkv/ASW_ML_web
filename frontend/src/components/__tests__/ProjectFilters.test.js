import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ProjectFilters from '../ProjectFilters';

// Mock data for testing
const mockProjects = ['6', '10', '12', '15', '20'];
const mockOnSelect = jest.fn();
const mockOnReset = jest.fn();

describe('ProjectFilters Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders project chips correctly', () => {
    render(
      <ProjectFilters
        projects={mockProjects}
        selectedProject={null}
        onSelect={mockOnSelect}
        onReset={mockOnReset}
      />
    );

    // Check if all project chips are rendered
    mockProjects.forEach(projectId => {
      expect(screen.getByText(`Project ${projectId}`)).toBeInTheDocument();
    });

    // Check if filter header is present
    expect(screen.getByText('Filter by Project')).toBeInTheDocument();
    expect(screen.getByText(`Showing all ${mockProjects.length} projects`)).toBeInTheDocument();
  });

  test('handles project selection', () => {
    render(
      <ProjectFilters
        projects={mockProjects}
        selectedProject={null}
        onSelect={mockOnSelect}
        onReset={mockOnReset}
      />
    );

    // Click on a project chip
    const projectChip = screen.getByText('Project 10');
    fireEvent.click(projectChip);

    expect(mockOnSelect).toHaveBeenCalledWith('10');
  });

  test('shows selected project state', () => {
    render(
      <ProjectFilters
        projects={mockProjects}
        selectedProject="10"
        onSelect={mockOnSelect}
        onReset={mockOnReset}
      />
    );

    // Check if selected project is highlighted
    expect(screen.getByText('Project 10')).toHaveClass('selected');
    expect(screen.getByText('10')).toBeInTheDocument(); // Badge showing selected project
    expect(screen.getByText('Showing data for Project 10')).toBeInTheDocument();
  });

  test('handles reset button click', () => {
    render(
      <ProjectFilters
        projects={mockProjects}
        selectedProject="10"
        onSelect={mockOnSelect}
        onReset={mockOnReset}
      />
    );

    // Click reset button
    const resetButton = screen.getByText('Show All');
    fireEvent.click(resetButton);

    expect(mockOnReset).toHaveBeenCalled();
  });

  test('handles keyboard navigation', () => {
    render(
      <ProjectFilters
        projects={mockProjects}
        selectedProject={null}
        onSelect={mockOnSelect}
        onReset={mockOnReset}
      />
    );

    const firstChip = screen.getByText('Project 6');
    firstChip.focus();

    // Test Enter key
    fireEvent.keyDown(firstChip, { key: 'Enter' });
    expect(mockOnSelect).toHaveBeenCalledWith('6');

    // Test Space key
    fireEvent.keyDown(firstChip, { key: ' ' });
    expect(mockOnSelect).toHaveBeenCalledWith('6');
  });

  test('handles disabled state', () => {
    render(
      <ProjectFilters
        projects={mockProjects}
        selectedProject={null}
        onSelect={mockOnSelect}
        onReset={mockOnReset}
        disabled={true}
      />
    );

    const projectChip = screen.getByText('Project 10');
    fireEvent.click(projectChip);

    // Should not call onSelect when disabled
    expect(mockOnSelect).not.toHaveBeenCalled();
  });

  test('renders nothing when no projects', () => {
    const { container } = render(
      <ProjectFilters
        projects={[]}
        selectedProject={null}
        onSelect={mockOnSelect}
        onReset={mockOnReset}
      />
    );

    expect(container.firstChild).toBeNull();
  });

  test('sorts projects numerically', () => {
    const unsortedProjects = ['20', '6', '15', '10', '12'];
    
    render(
      <ProjectFilters
        projects={unsortedProjects}
        selectedProject={null}
        onSelect={mockOnSelect}
        onReset={mockOnReset}
      />
    );

    // Get all project chips
    const chips = screen.getAllByText(/Project \d+/);
    
    // Check if they're sorted numerically
    expect(chips[0]).toHaveTextContent('Project 6');
    expect(chips[1]).toHaveTextContent('Project 10');
    expect(chips[2]).toHaveTextContent('Project 12');
    expect(chips[3]).toHaveTextContent('Project 15');
    expect(chips[4]).toHaveTextContent('Project 20');
  });
}); 