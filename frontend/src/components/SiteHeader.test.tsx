import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SiteHeader } from './SiteHeader'

describe('SiteHeader', () => {
  it('links the application brand to the home page with the decorative SVG logo', () => {
    render(<SiteHeader />)

    const homeLink = screen.getByRole('link', {
      name: 'Ir al inicio de VeriSon',
    })
    const logo = homeLink.querySelector('img')

    expect(homeLink).toHaveAttribute('href', '/')
    expect(homeLink).toContainElement(screen.getByText('VeriSon'))
    expect(logo).toHaveAttribute('src', '/logo.svg')
    expect(logo).toHaveAttribute('alt', '')
  })
})
