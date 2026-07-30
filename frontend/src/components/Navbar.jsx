function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <span className="navbar-logo">NeuroScope</span>
        <div className="navbar-links">
          <a href="#home">Home</a>
          <a href="#info">Info</a>
          <a href="https://github.com/" target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;