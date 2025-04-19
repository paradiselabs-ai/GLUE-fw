# Changelog

All notable changes to the GLUE framework will be documented in this file.


## [Unreleased]

### Added

- Model capability caching to avoid repeated API failures with models that don't support certain features


## [0.1.0-alpha] - 2025-04-02


### Added

- Core framework components with all tests passing
- Basic CLI functionality for running GLUE applications
- Multi-agent team management system
- Tool integration framework
- Support for multiple AI providers (OpenAI, Anthropic, etc.)
- Portkey.ai integration for API key management and usage tracking
- Basic documentation and examples


### Changed

- Improved error handling and logging
- Streamlined API for creating and managing teams
- Enhanced message routing between agents


### Fixed

- Flow class now properly handles task cancellation and event loop closure
- GlueApp close method ensures all flows are properly terminated
- Model class maintains backward compatibility with test cases


### Known Limitations

- Incomplete CLI features (some commands still under development)
- Limited documentation (will be expanded in future releases)
- Some APIs may change in future releases
- Limited number of built-in tools
