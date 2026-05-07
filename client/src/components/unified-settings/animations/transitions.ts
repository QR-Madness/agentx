/**
 * Framer Motion animation variants for unified settings
 */

export const backdropVariants = {
  initial: { opacity: 0, backdropFilter: 'blur(0px)' },
  animate: { opacity: 1, backdropFilter: 'blur(8px)' },
  exit: { opacity: 0, backdropFilter: 'blur(0px)' }
};

export const containerVariants = {
  initial: { scale: 0.95, opacity: 0, y: 20 },
  animate: {
    scale: 1,
    opacity: 1,
    y: 0,
    transition: {
      type: 'spring' as const,
      damping: 25,
      stiffness: 300
    }
  },
  exit: {
    scale: 0.95,
    opacity: 0,
    y: 20,
    transition: {
      duration: 0.2
    }
  }
};

export const navVariants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: {
      delay: 0.1,
      duration: 0.3
    }
  }
};

export const contentVariants = {
  initial: { opacity: 0, x: 20 },
  animate: {
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.3
    }
  },
  exit: {
    opacity: 0,
    x: -20,
    transition: {
      duration: 0.2
    }
  }
};

export const staggerChildrenVariants = {
  animate: {
    transition: {
      staggerChildren: 0.05
    }
  }
};

export const fadeInVariant = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 }
};
