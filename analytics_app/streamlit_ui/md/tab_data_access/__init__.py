"""
Tab-specific data access modules.

Organization:
    Each tab has its own subdirectory containing optimized data loaders.
    This keeps data access logic modular, maintainable, and performant.

Structure:
    tab_data_access/
    ├── homepage/       - Homepage tab data loaders
    ├── managers/       - Managers tab data loaders
    ├── players/        - Players tab data loaders
    ├── draft/          - Draft tab data loaders
    ├── transactions/   - Transactions tab data loaders
    ├── simulations/    - Simulations tab data loaders
    └── hall_of_fame/   - Hall of Fame tab data loaders

Pattern:
    Each subdirectory contains focused modules for specific data needs:
    - Column-specific queries (SELECT only needed columns)
    - Combined queries (reduce DB round-trips)
    - Lazy loading (load data when needed, not upfront)
"""
